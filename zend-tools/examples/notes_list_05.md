which notes start with the prefix 'meeting-'<|im_end|>
<|im_start|>assistant
<|projection|><think>
Okay, the user is asking which notes start with the prefix 'meeting-'. Let me check the available tools. There's a tool called notes_list that can browse stored notes by key prefix. The parameters include 'prefix' which is exactly what the user is asking for. The function requires the prefix parameter,<|projection|> so I should set that to 'meeting-'. The other parameters like max_results and tags are optional. Since the user didn't specify a number, I'll default max_results to 50. No need to include the tags parameter since there's no mention of tags in the query. So the tool call should be<|projection|> notes_list with prefix 'meeting-' and max_results 50.
</think>

<tool_call>
<|projection|>{"name": "notes_list", "arguments": {"prefix": "meeting-", "max_results": 50}}
</tool_call><|projection|>