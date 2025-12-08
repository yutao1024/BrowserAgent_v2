#!/bin/bash
# Tool Server for data_generate_conclusion.py
# Port: 30810

# Stop existing ray instance
ray stop

# Start new ray instance
ray start --head --dashboard-host=0.0.0.0

host=0.0.0.0
port=30816
tool_server_url=http://$host:$port/get_observation

echo "=================================="
echo "Starting Tool Server"
echo "Port: $port"
echo "URL: $tool_server_url"
echo "=================================="

# Start the server
cd /data/yutao/browseragent2_dev/BrowserAgent/verl-tool
python -m verl_tool.servers.serve --host $host --port $port --tool_type "text_browser"







