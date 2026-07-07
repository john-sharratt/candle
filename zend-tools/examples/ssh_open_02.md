connect over SSH to deploy@example.com using credential deploy-key<|im_end|>
<|im_start|>assistant
<|projection|><think>
Okay, the user wants to connect via SSH to deploy@example.com using the credential named deploy-key. Let me check the tools available.

The ssh_open tool is for opening an SSH session. The parameters needed are credential_name, host, port, and idle_timeout_sec. The user specified the credential as deploy-key<|projection|>, so that's the credential_name. The host is example.com, since the email is deploy@example.com. The port isn't mentioned, so I'll default it to 22. The idle timeout can be left as the default of 900 seconds.

I need to make sure the host is just the<|projection|> hostname or IP, not including the username. So, host should be "example.com". The credential's default port is probably 22, so unless specified, that's okay. 

So the tool call should be ssh_open with credential_name "deploy-key", host "example.com", and port 22.<|projection|> The other parameters are optional and can be omitted. Let me structure the JSON accordingly.
</think>

<tool_call>
<|projection|>{"name": "ssh_open", "arguments": {"credential_name": "deploy-key", "host": "example.com", "port": 22}}
</tool_call><|projection|>