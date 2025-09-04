from llama_index.core.workflow import Event, StartEvent, StopEvent, Workflow, step
from llama_index.core import PromptTemplate
from llama_index.core.llms import ChatMessage
from pydantic import BaseModel, Field

system_prompt = """
You have researched a specific AI developer tool which could be a tool, platform, library, framework, language or model (among others). You will output information about this in this specific format:

Here are some further instructions;
Don’t use over the top or dramatic language like groundbreaking, mindblowing etc.
Don’t use strong terms like “amazing”, “intriguing”, “fascinating”, “must-read”, or “outstanding”.
Do NOT use these words: “embarking”, “delve”, “vibrant”, “realm”, “endeavour” or “dive into”.

Fields should only be filled if the information is available in the following context. Otherwise, leave the field empty. 
Here are the source materials from you research to use:
{context}
"""

class ToolInformationOutput(BaseModel):
    description: str = Field(..., description="Give a 4 sentence description of the tool in one full paragraph.")
    parent: str = Field(..., description="Give a comma separated list of direct parent software tools it uses / is dependent on.")
    languages: list[str] = Field(..., description="Give a comma separated list of compatible software languages.")
    mapping: list[str] = Field(..., description="Give a comma separated list of alternative spellings of this tool. Include acronyms (or full words if the acronym was given for the tool name)")
    category: list[str] = Field(..., description="Give a comma separated list of the types of task the product is used for. Use only 1-3 word category names not full complex sentences.")
    related_tools: list[str] = Field(..., description="What other tools is it compatible with? Give a comma separated list of the types of task the product is used for. Use only 1-3 word category names not full complex sentences.")
    company: str = Field(..., description="What company or organisation owns/builds the tool?")


class ContextEvent(Event):
    context: str

class ToolAnalysisWorkflow(Workflow):
    def __init__(self, openai_llm):
        super().__init__()
        self.openai_llm = openai_llm
        self.prompt_template = PromptTemplate(template=system_prompt)
        self.structured_llm = openai_llm.as_structured_llm(ToolInformationOutput)

    @step
    async def format_context(self, ev: StartEvent) -> ContextEvent:
        """Format the context using the prompt template."""
        formatted_prompt = self.prompt_template.format(context=ev.context)
        return ContextEvent(context=formatted_prompt)

    @step
    async def generate_structured_output(self, ev: ContextEvent) -> StopEvent:
        """Generate structured output using the LLM."""
        try:
            # Convert string to ChatMessage format for achat
            messages = [ChatMessage(role="user", content=ev.context)]
            result = await self.structured_llm.achat(messages)

            # Extract the structured object from the response
            if hasattr(result, 'message') and hasattr(result.message, 'content'):
                # If content is already a structured object, return it
                if isinstance(result.message.content, ToolInformationOutput):
                    return StopEvent(result=result.message.content)
                else:
                    # If it's still a string, try to parse it
                    return StopEvent(result=result.message.content)
            else:
                return StopEvent(result=result)

        except Exception as e:
            # Fallback to synchronous structured predict
            result = self.structured_llm.predict(ev.context)
            return StopEvent(result=result)

