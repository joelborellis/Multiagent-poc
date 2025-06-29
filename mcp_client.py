from mcp.client.streamable_http import streamablehttp_client
from mcp import ClientSession
import asyncio


async def main():
    #url="https://echo-mcp.purplesand-ac2cef21.eastus2.azurecontainerapps.io/echo/mcp"
    url="http://localhost:8000/mcp"
    # Connect to a streamable HTTP server
    async with streamablehttp_client(url=url) as (
        read_stream,
        write_stream,
        _,
    ):
        # Create a session using the client streams
        async with ClientSession(read_stream, write_stream) as session:
            # Initialize the connection
            await session.initialize()
            
            print("getting session")
            # List available tools
            tools_result = await session.list_tools()
            print("Available tools:")
            for tool in tools_result.tools:
                print(f"  - {tool.name}: {tool.description}")

            # Call our calculator tool
            result = await session.call_tool("echo", {"message": "This is a call to the echo tool"})
            print(f"{result.content[0].text}")
            
if __name__ == "__main__":
    asyncio.run(main())