from mcp.server.fastmcp import FastMCP

# Stateless server (no session persistence)
mcp = FastMCP("StatelessServer", stateless_http=True)

@mcp.tool(description="A simple echo tool")
def echo(message: str) -> str:
    return f"Echo: {message}"


if __name__ == "__main__":
    # Initialize and run the server
    mcp.run(transport='streamable-http')