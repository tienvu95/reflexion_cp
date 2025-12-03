import re, string, os
from typing import List, Union, Literal, Any, Optional
from enum import Enum
try:
    import tiktoken
except Exception:
    tiktoken = None
try:
    from langchain.llms.base import BaseLLM
    from langchain.chat_models.base import BaseChatModel
    from langchain.schema import (
        SystemMessage,
        HumanMessage,
        AIMessage,
    )
except Exception:
    # Try package-aware shim import first (when running as a package/module),
    # then fall back to top-level shim (when running from repo root).
    try:
        from hotpotqa_runs.langchain_shim import BaseLLM, BaseChatModel, SystemMessage, HumanMessage, AIMessage
    except Exception:
        from langchain_shim import BaseLLM, BaseChatModel, SystemMessage, HumanMessage, AIMessage
# Try to import AnyOpenAILLM from the package or top-level fallback.
try:
    from hotpotqa_runs.llm import AnyOpenAILLM
except Exception:
    try:
        from llm import AnyOpenAILLM
    except Exception:
        AnyOpenAILLM = None
# Docstore integration is optional. We avoid importing heavy langchain docstore
# modules at top-level to make the module importable when langchain is not installed.
# package-aware imports: allow running as `python -m hotpotqa_runs.run_local_llm_demo`
try:
    from hotpotqa_runs.prompt_shim import PromptTemplate
    from hotpotqa_runs.prompts import (
        reflect_prompt,
        react_agent_prompt,
        react_reflect_agent_prompt,
        REFLECTION_HEADER,
        LAST_TRIAL_HEADER,
        REFLECTION_AFTER_LAST_TRIAL_HEADER,
        cot_agent_prompt,
        cot_reflect_agent_prompt,
        cot_reflect_prompt,
        COT_INSTRUCTION,
        COT_REFLECT_INSTRUCTION,
    )
    from hotpotqa_runs.fewshots import WEBTHINK_SIMPLE6, REFLECTIONS, COT, COT_REFLECT
except Exception:
    from prompt_shim import PromptTemplate
    from prompts import (
        reflect_prompt,
        react_agent_prompt,
        react_reflect_agent_prompt,
        REFLECTION_HEADER,
        LAST_TRIAL_HEADER,
        REFLECTION_AFTER_LAST_TRIAL_HEADER,
        cot_agent_prompt,
        cot_reflect_agent_prompt,
        cot_reflect_prompt,
        COT_INSTRUCTION,
        COT_REFLECT_INSTRUCTION,
    )
    from fewshots import WEBTHINK_SIMPLE6, REFLECTIONS, COT, COT_REFLECT


class ReflexionStrategy(Enum):
    """
    NONE: No reflection
    LAST_ATTEMPT: Use last reasoning trace in context 
    REFLEXION: Apply reflexion to the next reasoning trace 
    LAST_ATTEMPT_AND_REFLEXION: Use last reasoning trace in context and apply reflexion to the next reasoning trace 
    """
    NONE = 'base'
    LAST_ATTEMPT = 'last_trial' 
    REFLEXION = 'reflexion'
    LAST_ATTEMPT_AND_REFLEXION = 'last_trial_and_reflexion'


class CoTAgent:
    def __init__(self,
                    question: str,
                    context: str,
                    key: str,
                    agent_prompt: PromptTemplate = cot_reflect_agent_prompt,
                    reflect_prompt: PromptTemplate = cot_reflect_prompt,
                    cot_examples: str = COT,
                    reflect_examples: str = COT_REFLECT,
                    self_reflect_llm: Optional[Any] = None,
                    action_llm: Optional[Any] = None,
                    force_finish_format: bool = False,
                    ) -> None:
        self.question = question
        self.context = context
        self.key = key
        self.agent_prompt = agent_prompt
        self.reflect_prompt = reflect_prompt
        self.cot_examples = cot_examples 
        self.reflect_examples = reflect_examples
        self.self_reflect_llm = self_reflect_llm
        self.action_llm = action_llm
        self.force_finish_format = force_finish_format
        self._debug_enabled: bool = False
        self.reflections: List[str] = []
        self.reflections_str = ''
        self.answer = ''
        self.step_n: int = 0
        self.reset()

    def run(self,
            reflexion_strategy: ReflexionStrategy = ReflexionStrategy.REFLEXION) -> None:
        if self.step_n > 0 and not self.is_correct() and reflexion_strategy != ReflexionStrategy.NONE:
            self.reflect(reflexion_strategy)
        self.reset()
        self.step()
        self.step_n += 1

        # If no LLMs provided, try to lazily construct OpenAI wrappers (only if API key available).
        if self.self_reflect_llm is None or self.action_llm is None:
            try:
                from langchain import OpenAI
                if 'OPENAI_API_KEY' in os.environ:
                    if self.self_reflect_llm is None:
                        self.self_reflect_llm = OpenAI(temperature=0, max_tokens=250, model_name="gpt-3.5-turbo", openai_api_key=os.environ['OPENAI_API_KEY'])
                    if self.action_llm is None:
                        self.action_llm = OpenAI(temperature=0, max_tokens=250, model_name="gpt-3.5-turbo", openai_api_key=os.environ['OPENAI_API_KEY'])
            except Exception:
                # No OpenAI available; leave as None and raise clear errors at call time.
                pass
    
    def reflect(self,
                strategy: ReflexionStrategy) -> None:
        print('Running Reflexion strategy...')
        if strategy == ReflexionStrategy.LAST_ATTEMPT:
            self.reflections = [self.scratchpad]
            self.reflections_str = format_last_attempt(self.question , self.reflections[0])
        elif strategy == ReflexionStrategy.REFLEXION:
            self.reflections += [self.prompt_reflection()]
            self.reflections_str = format_reflections(self.reflections)
        elif strategy == ReflexionStrategy.LAST_ATTEMPT_AND_REFLEXION:
            self.reflections_str = format_last_attempt(self.question , self.scratchpad)
            self.reflections = [self.prompt_reflection()]
            self.reflections_str += '\n'+ format_reflections(self.reflections, header = REFLECTION_AFTER_LAST_TRIAL_HEADER)
        else:
            raise NotImplementedError(f'Unknown reflection strategy: {strategy}')
        print(self.reflections_str)
    
    def prompt_reflection(self) -> str:
        if self.self_reflect_llm is None:
            raise RuntimeError("No reflection LLM configured. Pass `self_reflect_llm` (e.g. AnyHFLLM) or set OPENAI_API_KEY for a fallback OpenAI LLM.")
        return format_step(self.self_reflect_llm(self._build_reflection_prompt()))

    def reset(self) -> None:
        self.scratchpad: str = ''
        self.finished = False

    def step(self) -> None:
        # Think
        self.scratchpad += f'\nThought:'
        self.scratchpad += ' ' + self.prompt_agent()
        if self._debug_enabled:
            print(self.scratchpad.split('\n')[-1])

        # Act
        self.scratchpad += f'\nAction:'
        action = self.prompt_agent()
        action_type, argument = parse_action(action)
        clean_action = action if action_type != 'Finish' else f'Finish[{argument}]'
        self.scratchpad += ' ' + clean_action
        if self._debug_enabled:
            print(self.scratchpad.split('\n')[-1])

        # Observe
        self.scratchpad += f'\nObservation: '
        if action_type == 'Finish':
            self.answer = argument
            if self.is_correct():
                self.scratchpad += 'Answer is CORRECT'
            else:
                self.scratchpad += 'Answer is INCORRECT'
            reason = self._generate_reason(argument)
            if reason:
                self.scratchpad += f'\n{reason}'
            self.finished = True
            return
        else:
            print('Invalid action type, please try again.')

    def prompt_agent(self) -> str:
        if self.action_llm is None:
            raise RuntimeError("No action LLM configured. Pass `action_llm` (e.g. AnyHFLLM) or set OPENAI_API_KEY for a fallback OpenAI LLM.")
        # Try to get a properly formatted Action[...] from the LLM. If the
        # model returns freeform text, retry with a brief instruction to
        # respond only with the Action[...] line. This helps when models do
        # not strictly follow the examples.
        max_retries = 2
        attempt = 0
        last_out = ''
        while attempt <= max_retries:
            out = format_step(self.action_llm(self._build_agent_prompt()))
            # sanitize raw LLM output to remove training artifacts before parsing
            out = _clean_agent_output(out)
            last_out = out
            action_type, argument = parse_action(out)
            if action_type is not None:
                return out
            # prepare a short follow-up that requests the correct format
            followup = '\nPlease respond with exactly one `Action:` line in the format Action[<type>[<argument>]] using one of: Search[...], Lookup[...], Finish[...]. Output only that Action line.'
            if self.force_finish_format:
                followup = '\nWhen you decide to finish, respond with exactly one `Action:` line in the format Action: Finish[yes] or Action: Finish[no] or Action: Finish[maybe]. Do not output any other text.\n' + followup
            out = format_step(self.action_llm(self._build_agent_prompt() + followup))
            out = _clean_agent_output(out)
            action_type, argument = parse_action(out)
            if action_type is not None:
                return out
            attempt += 1
        # return last output even if malformed; caller will detect invalid action
        return last_out

    def _generate_reason(self, label: str) -> str:
        llm = self.self_reflect_llm or self.action_llm
        prompt = (
            "You are validating a PubMedQA answer. Provide exactly one sentence beginning with 'Reason:' that cites key evidence from the abstract supporting the label.\n"
            f"Question: {self.question}\n"
            f"Abstract: {self.context}\n"
            f"Answer label: {label}\n"
            "Reason: "
        )
        try:
            out = llm(prompt)
            out = out.strip()
            if 'Reason:' not in out:
                out = 'Reason: ' + out
            reason = out.split('\n', 1)[0]
            reason = reason.split('.', 1)[0].strip()
            if not reason.endswith('.'):
                reason += '.'
            return reason
        except Exception:
            return f"Reason: {label} based on the evidence described above."
    
    def _build_agent_prompt(self) -> str:
        return self.agent_prompt.format(
                            examples = self.cot_examples,
                            reflections = self.reflections_str,
                            context = self.context,
                            question = self.question,
                            scratchpad = self.scratchpad)

    def _build_reflection_prompt(self) -> str:
        return self.reflect_prompt.format(
                            examples = self.reflect_examples,
                            context = self.context,
                            question = self.question,
                            scratchpad = self.scratchpad)
 
    def is_finished(self) -> bool:
        return self.finished

    def is_correct(self) -> bool:
        return EM(self.answer, self.key)   

class ReactAgent:
    def __init__(self,
                 question: str,
                 key: str,
                 max_steps: int = 6,
                 agent_prompt: PromptTemplate = react_agent_prompt,
                 docstore: Optional[Any] = None,
                 react_llm: Optional[Any] = None,
                 force_finish_format: bool = False,
                 ) -> None:
        
        self.question = question
        self.answer = ''
        self.key = key
        self.max_steps = max_steps
        self.agent_prompt = agent_prompt
        self.react_examples = WEBTHINK_SIMPLE6
        self._debug_enabled: bool = False

        # If a docstore was provided, try to wrap it with langchain's DocstoreExplorer
        if docstore is None:
            self.docstore = None
        else:
            try:
                from langchain.agents.react.base import DocstoreExplorer
                self.docstore = DocstoreExplorer(docstore)
            except Exception:
                # LangChain wrapping failed — fall back to using provided docstore
                # directly (as long as it implements .search() and .lookup()).
                try:
                    import traceback
                    print('Warning: DocstoreExplorer wrapping failed, falling back to provided docstore. Exception:')
                    traceback.print_exc()
                except Exception:
                    pass
        self.docstore = docstore
        self.llm = react_llm
        self.force_finish_format = force_finish_format
        self._debug_enabled = False

        # tolerant tokenizer: try tiktoken, otherwise simple whitespace encoder
        try:
            if tiktoken is not None:
                self.enc = tiktoken.encoding_for_model("text-davinci-003")
            else:
                raise Exception()
        except Exception:
            class _SimpleEnc:
                def encode(self, s: str):
                    return s.split()
            self.enc = _SimpleEnc()

        self.__reset_agent()

    def run(self, reset = True) -> None:
        if reset:
            self.__reset_agent()
        
        while not self.is_halted() and not self.is_finished():
            self.step()
    
    def step(self) -> None:
        # Think
        self.scratchpad += f'\nThought {self.step_n}:'
        self.scratchpad += ' ' + self.prompt_agent()
        if self._debug_enabled:
            print(self.scratchpad.split('\n')[-1])

        # Act
        self.scratchpad += f'\nAction {self.step_n}:'
        action = self.prompt_agent()
        action_type, argument = parse_action(action)
        clean_action = action if action_type != 'Finish' else f'Finish[{argument}]'
        self.scratchpad += ' ' + clean_action
        if self._debug_enabled:
            print(self.scratchpad.split('\n')[-1])

        # Observe
        self.scratchpad += f'\nObservation {self.step_n}: '
        
        if action_type == 'Finish':
            self.answer = argument
            if self.is_correct():
                self.scratchpad += 'Answer is CORRECT'
            else: 
                self.scratchpad += 'Answer is INCORRECT'
            reason = self._generate_reason(argument)
            if reason:
                self.scratchpad += f'\n{reason}'
            self.finished = True
            self.step_n += 1
            return

        if action_type == 'Search':
            # Debug: report docstore presence and type
            try:
                if self.docstore is None:
                    if self._debug_enabled:
                        print('DEBUG: ReactAgent.step - docstore is None when handling Search[{}]'.format(argument))
                    self.scratchpad += ' [No docstore configured]'
                else:
                    try:
                        if self._debug_enabled:
                            print('DEBUG: ReactAgent.step - calling docstore.search; docstore type:', type(self.docstore))
                    except Exception:
                        pass
                    try:
                        self.scratchpad += format_step(self.docstore.search(argument))
                    except Exception as e:
                        import traceback
                        print('DEBUG: docstore.search raised exception:')
                        traceback.print_exc()
                        self.scratchpad += f'Could not find that page, please try again.'
            except Exception:
                # Defensive: ensure agent still continues even if debug printing fails
                if self.docstore is None:
                    self.scratchpad += ' [No docstore configured]'
                else:
                    try:
                        self.scratchpad += format_step(self.docstore.search(argument))
                    except Exception:
                        self.scratchpad += f'Could not find that page, please try again.'
            
        elif action_type == 'Lookup':
            try:
                if self.docstore is None:
                    if self._debug_enabled:
                        print('DEBUG: ReactAgent.step - docstore is None when handling Lookup[{}]'.format(argument))
                    self.scratchpad += ' [No docstore configured] '
                else:
                    try:
                        if self._debug_enabled:
                            print('DEBUG: ReactAgent.step - calling docstore.lookup; docstore type:', type(self.docstore))
                    except Exception:
                        pass
                    try:
                        self.scratchpad += format_step(self.docstore.lookup(argument))
                    except ValueError:
                        self.scratchpad += f'The last page Searched was not found, so you cannot Lookup a keyword in it. Please try one of the similar pages given.'
                    except Exception:
                        import traceback
                        print('DEBUG: docstore.lookup raised exception:')
                        traceback.print_exc()
                        self.scratchpad += f'Could not lookup that term in the last page.'
            except Exception:
                # Defensive
                if self.docstore is None:
                    self.scratchpad += ' [No docstore configured] '
                else:
                    try:
                        self.scratchpad += format_step(self.docstore.lookup(argument))
                    except Exception:
                        self.scratchpad += f'Could not lookup that term in the last page.'

        else:
            self.scratchpad += 'Invalid Action. Valid Actions are Lookup[<topic>] Search[<topic>] and Finish[<answer>].'

        print(self.scratchpad.split('\n')[-1])

        self.step_n += 1

    def prompt_agent(self) -> str:
        if self.llm is None:
            raise RuntimeError("No LLM configured for ReactAgent. Pass `react_llm` (e.g. AnyHFLLM) or set OPENAI_API_KEY for a fallback OpenAI LLM.")
        # Similar retry wrapper as CoTAgent.prompt_agent to enforce Action[...] format
        max_retries = 2
        attempt = 0
        last_out = ''
        while attempt <= max_retries:
            out = format_step(self.llm(self._build_agent_prompt()))
            out = _clean_agent_output(out)
            last_out = out
            action_type, argument = parse_action(out)
            if action_type is not None:
                return out
            followup = '\nPlease respond with exactly one `Action:` line in the format Action[<type>[<argument>]] using one of: Search[...], Lookup[...], Finish[...]. Output only that Action line.'
            if self.force_finish_format:
                followup = '\nWhen you decide to finish, respond with exactly one `Action:` line in the format Action: Finish[yes] or Action: Finish[no] or Action: Finish[maybe]. Do not output any other text.\n' + followup
            out = format_step(self.llm(self._build_agent_prompt() + followup))
            out = _clean_agent_output(out)
            action_type, argument = parse_action(out)
            if action_type is not None:
                return out
            attempt += 1
        return last_out
    
    def _build_agent_prompt(self) -> str:
        return self.agent_prompt.format(
                            examples = self.react_examples,
                            question = self.question,
                            scratchpad = self.scratchpad)
    
    def is_finished(self) -> bool:
        return self.finished

    def is_correct(self) -> bool:
        return EM(self.answer, self.key)

    def is_halted(self) -> bool:
        return ((self.step_n > self.max_steps) or (len(self.enc.encode(self._build_agent_prompt())) > 3896)) and not self.finished

    def __reset_agent(self) -> None:
        self.step_n = 1
        self.finished = False
        self.scratchpad: str = ''

    def set_qa(self, question: str, key: str) -> None:
        self.question = question
        self.key = key

    def _generate_reason(self, label: str) -> str:
        llm = self.llm
        if llm is None:
            return f"Reason: {label}."
        prompt = (
            "You are validating a PubMedQA answer. Provide exactly one sentence beginning with 'Reason:' using the retrieved evidence below.\n"
            f"Question: {self.question}\n"
            f"Retrieved evidence:\n{context_snip}\n"
            f"Answer label: {label}\n"
            "Reason: "
        )
        try:
            out = llm(prompt)
            out = out.strip()
            if 'Reason:' not in out:
                out = 'Reason: ' + out
            reason = out.split('\n', 1)[0]
            reason = reason.split('.', 1)[0].strip()
            if not reason.endswith('.'):
                reason += '.'
            return reason
        except Exception:
            return f"Reason: {label}."

class ReactReflectAgent(ReactAgent):
    def __init__(self,
                 question: str,
                 key: str,
                 max_steps: int = 6,
                 agent_prompt: PromptTemplate = react_reflect_agent_prompt,
                 reflect_prompt: PromptTemplate = reflect_prompt,
                 docstore: Optional[Any] = None,
                 react_llm: Optional[Any] = None,
                 reflect_llm: Optional[Any] = None,
                 force_finish_format: bool = False,
                 ) -> None:
        super().__init__(question, key, max_steps, agent_prompt, docstore, react_llm, force_finish_format=force_finish_format)
        self.reflect_llm = reflect_llm
        self.reflect_prompt = reflect_prompt
        self.reflect_examples = REFLECTIONS
        self.reflections: List[str] = []
        self.reflections_str: str = ''
    
    def run(self, reset = True, reflect_strategy: ReflexionStrategy = ReflexionStrategy.REFLEXION) -> None:
        if (self.is_finished() or self.is_halted()) and not self.is_correct():
            self.reflect(reflect_strategy)

        ReactAgent.run(self, reset)
    
    def reflect(self,
                strategy: ReflexionStrategy) -> None:
        print('Reflecting...')
        if strategy == ReflexionStrategy.LAST_ATTEMPT:
            self.reflections = [self.scratchpad]
            self.reflections_str = format_last_attempt(self.question, self.reflections[0])
        elif strategy == ReflexionStrategy.REFLEXION: 
            self.reflections += [self.prompt_reflection()]
            self.reflections_str = format_reflections(self.reflections)
        elif strategy == ReflexionStrategy.LAST_ATTEMPT_AND_REFLEXION: 
            self.reflections_str = format_last_attempt(self.question, self.scratchpad)
            self.reflections = [self.prompt_reflection()]
            self.reflections_str += format_reflections(self.reflections, header = REFLECTION_AFTER_LAST_TRIAL_HEADER)
        else:
            raise NotImplementedError(f'Unknown reflection strategy: {strategy}')
        print(self.reflections_str)
    
    def prompt_reflection(self) -> str:
        if self.reflect_llm is None:
            raise RuntimeError("No reflection LLM configured for ReactReflectAgent. Pass `reflect_llm` (e.g. AnyHFLLM) or set OPENAI_API_KEY for a fallback OpenAI LLM.")
        return format_step(self.reflect_llm(self._build_reflection_prompt()))


    def _build_reflection_prompt(self) -> str:
        return self.reflect_prompt.format(
                            examples = self.reflect_examples,
                            question = self.question,
                            scratchpad = truncate_scratchpad(self.scratchpad, tokenizer=self.enc))
 
    def _build_agent_prompt(self) -> str:
        return self.agent_prompt.format(
                            examples = self.react_examples,
                            reflections = self.reflections_str,
                            question = self.question,
                            scratchpad = self.scratchpad)
   

### String Stuff ###
if tiktoken is not None:
    gpt2_enc = tiktoken.encoding_for_model("text-davinci-003")
else:
    class _SimpleEnc2:
        def encode(self, s: str):
            return s.split()
    gpt2_enc = _SimpleEnc2()

def parse_action(string):
    if string is None:
        return None, None
    s = string.strip()
    # If model printed an Action: prefix, take the text after it
    if s.lower().startswith('action:'):
        s = s[len('action:'):].strip()

    # Try strict pattern first: TYPE[arg]
    pattern = r'^(\w+)\[(.+)\]$'
    match = re.match(pattern, s)
    if match:
        action_type = match.group(1)
        argument = match.group(2)
        return action_type, argument

    # Try to find any TYPE[...] anywhere in the string
    match = re.search(r'(\w+)\[([^\]]+)\]', s)
    if match:
        return match.group(1), match.group(2)

    # Heuristic fallbacks: if the output is short, treat it as Finish[<output>]
    # Reject obvious training artifacts or repeated tokens that are not actions
    if re.search(r'END OF EXERCISE', s, flags=re.IGNORECASE):
        return None, None
    # Avoid accepting noisy ALL-CAPS output as a Finish
    if s.isupper() and len(s.split()) > 1:
        return None, None

    tokens = s.split()
    if 0 < len(tokens) <= 3:
        # Only accept short, explicit yes/no/maybe forms as Finish
        low = s.strip().lower()
        if low in ('yes', 'y', 'no', 'n', 'maybe', 'possibly', 'could', 'likely', 'uncertain', 'unsure', 'unclear'):
            return 'Finish', low
        # Also accept short phrases like 'no change' or 'not sure' conservatively
        if len(tokens) == 2 and any(t in ('no','not','unsure','unclear','maybe') for t in map(str.lower, tokens)):
            return 'Finish', s
    # Do not heuristically accept longer freeform outputs as Finish; require explicit Finish[...] or bracketed form

    # If the output contains the word 'finish' followed by bracket-like text
    m = re.search(r'finish\s*[:\[]\s*([^\]\n]+)', s, flags=re.IGNORECASE)
    if m:
        return 'Finish', m.group(1).strip()

    return None, None


def _clean_agent_output(out: str) -> str:
    """Sanitize raw agent/LLM output to remove common training artifacts.

    - Collapse repeated 'END OF EXERCISE' markers
    - Remove stray trailing 'END OF EXERCISE' fragments that confuse parsers
    - Strip excessive whitespace and control characters
    Returns cleaned string.
    """
    if out is None:
        return ''
    try:
        import re
        s = out.replace('\r', ' ')
        # collapse many repeated 'END OF EXERCISE' occurrences to a single marker
        s = re.sub(r'(END OF EXERCISE\.?\s*){2,}', 'END OF EXERCISE. ', s, flags=re.IGNORECASE)
        # remove long runs entirely
        s = re.sub(r'(END OF EXERCISE\.?\s*){5,}', '', s, flags=re.IGNORECASE)
        # if line begins with Action: and contains only END OF EXERCISE tokens, strip them
        s = re.sub(r'(?im)^(\s*Action:\s*)(END OF EXERCISE\.?\s*)+$', '\1', s)
        # strip excessive whitespace
        s = re.sub(r'\s{3,}', ' ', s)
        return s.strip()
    except Exception:
        return out.strip() if out else ''

def format_step(step: str) -> str:
    return step.replace('\r', '').strip()

def _needs_reason(text: str) -> bool:
    for line in text.splitlines():
        if line.strip().lower().startswith('reason:'):
            return False
    return True

def format_reflections(reflections: List[str],
                        header: str = REFLECTION_HEADER) -> str:
    if reflections == []:
        return ''
    else:
        return header + 'Reflections:\n- ' + '\n- '.join([r.strip() for r in reflections])

def format_last_attempt(question: str,
                        scratchpad: str,
                        header: str = LAST_TRIAL_HEADER):
    return header + f'Question: {question}\n' + truncate_scratchpad(scratchpad, tokenizer=gpt2_enc).strip('\n').strip() + '\n(END PREVIOUS TRIAL)\n'

def truncate_scratchpad(scratchpad: str, n_tokens: int = 1600, tokenizer = gpt2_enc) -> str:
    lines = scratchpad.split('\n')
    observations = filter(lambda x: x.startswith('Observation'), lines)
    observations_by_tokens = sorted(observations, key=lambda x: len(tokenizer.encode(x)))
    while len(gpt2_enc.encode('\n'.join(lines))) > n_tokens:
        largest_observation = observations_by_tokens.pop(-1)
        ind = lines.index(largest_observation)
        lines[ind] = largest_observation.split(':')[0] + ': [truncated wikipedia excerpt]'
    return '\n'.join(lines)

def normalize_answer(s):
  def remove_articles(text):
    return re.sub(r"\b(a|an|the)\b", " ", text)
  
  def white_space_fix(text):
      return " ".join(text.split())

  def remove_punc(text):
      exclude = set(string.punctuation)
      return "".join(ch for ch in text if ch not in exclude)

  def lower(text):
      return text.lower()

  return white_space_fix(remove_articles(remove_punc(lower(s))))

def EM(answer, key) -> bool:
    return normalize_answer(answer) == normalize_answer(key)
