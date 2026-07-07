delete the saved credential named old-api-key<|im_end|>
<|im_start|>assistant
<|projection|><think>
Okay, the user wants to delete a credential called "old-api-key". Let me check the tools available. There's a function called credential_delete that takes a name parameter. The description says it permanently deletes a stored credential by name. The user provided the exact name, so I can use this tool. I<|projection|> need to make sure the arguments are correctly formatted as JSON inside the tool_call tags. No other tools are needed here. Just call credential_delete with the name "old-api-key".
</think>

<tool_call>
<|projection|>{"name": "credential_delete", "arguments": {"name": "old-api-key"}}
</tool_call><|projection|>