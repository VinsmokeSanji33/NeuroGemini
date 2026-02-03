"""MCP Server for Synthetic Cortex tools.

Exposes tools for agent interaction:
- toggle_resolution: Switch camera between low/high resolution
- spatial_audio_trigger: Play directional audio cue
- emergency_halt: Immediate system stop
- ask_user_clarification: Audio prompt when CAIR < threshold
"""

import os
import logging
import asyncio
from typing import Any
from contextlib import asynccontextmanager

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

logger = logging.getLogger(__name__)


class SyntheticCortexMCPServer:
    """MCP Server exposing Synthetic Cortex control tools."""
    
    def __init__(self):
        self.server = Server("synthetic-cortex")
        self._resolution_mode = "low"
        self._halt_flag = False
        self._clarification_callback = None
        self._audio_callback = None
        
        self._register_tools()
        
        logger.info("MCP Server initialized")
    
    def _register_tools(self) -> None:
        """Register all available tools."""
        
        @self.server.list_tools()
        async def list_tools() -> list[Tool]:
            return [
                Tool(
                    name="toggle_resolution",
                    description="Switch camera between low and high resolution modes. "
                               "Low resolution (640x480) for fast reflex processing, "
                               "high resolution (1280x720) for detailed cerebral analysis.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "mode": {
                                "type": "string",
                                "enum": ["low", "high"],
                                "description": "Resolution mode to switch to",
                            },
                        },
                        "required": ["mode"],
                    },
                ),
                Tool(
                    name="spatial_audio_trigger",
                    description="Trigger a spatial audio cue to alert the user. "
                               "Position determines stereo panning, urgency affects volume.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "position": {
                                "type": "number",
                                "minimum": -1.0,
                                "maximum": 1.0,
                                "description": "Horizontal position (-1=left, 0=center, 1=right)",
                            },
                            "urgency": {
                                "type": "string",
                                "enum": ["low", "medium", "high", "critical"],
                                "description": "Urgency level affecting volume and tone",
                            },
                            "object_type": {
                                "type": "string",
                                "description": "Type of object detected (affects tone frequency)",
                            },
                            "message": {
                                "type": "string",
                                "description": "Optional message to speak via TTS",
                            },
                        },
                        "required": ["position", "urgency"],
                    },
                ),
                Tool(
                    name="emergency_halt",
                    description="Trigger immediate system halt. Use when a critical hazard "
                               "is detected or user provides negative feedback. "
                               "Clears all state and stops processing.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "reason": {
                                "type": "string",
                                "description": "Reason for emergency halt",
                            },
                        },
                        "required": ["reason"],
                    },
                ),
                Tool(
                    name="ask_user_clarification",
                    description="Request user clarification via spatial audio when CAIR "
                               "score is below threshold. Pauses autonomous action until "
                               "user responds.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "question": {
                                "type": "string",
                                "description": "Question to ask the user",
                            },
                            "context": {
                                "type": "string",
                                "description": "Additional context about the detection",
                            },
                            "confidence": {
                                "type": "number",
                                "minimum": 0.0,
                                "maximum": 1.0,
                                "description": "Current confidence score",
                            },
                            "detection_bbox": {
                                "type": "array",
                                "items": {"type": "number"},
                                "description": "Bounding box [x1, y1, x2, y2] for spatial audio",
                            },
                        },
                        "required": ["question"],
                    },
                ),
            ]
        
        @self.server.call_tool()
        async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
            try:
                if name == "toggle_resolution":
                    return await self._handle_toggle_resolution(arguments)
                elif name == "spatial_audio_trigger":
                    return await self._handle_spatial_audio(arguments)
                elif name == "emergency_halt":
                    return await self._handle_emergency_halt(arguments)
                elif name == "ask_user_clarification":
                    return await self._handle_ask_clarification(arguments)
                else:
                    return [TextContent(type="text", text=f"Unknown tool: {name}")]
            except Exception as e:
                logger.error(f"Tool error: {e}")
                return [TextContent(type="text", text=f"Error: {str(e)}")]
    
    async def _handle_toggle_resolution(
        self,
        arguments: dict[str, Any],
    ) -> list[TextContent]:
        """Handle resolution toggle request."""
        mode = arguments.get("mode", "low")
        
        if mode not in ("low", "high"):
            return [TextContent(type="text", text=f"Invalid mode: {mode}")]
        
        old_mode = self._resolution_mode
        self._resolution_mode = mode
        
        logger.info(f"Resolution changed: {old_mode} -> {mode}")
        
        return [TextContent(
            type="text",
            text=f"Resolution mode changed to {mode}. "
                 f"{'High-detail cerebral analysis enabled.' if mode == 'high' else 'Fast reflex processing enabled.'}",
        )]
    
    async def _handle_spatial_audio(
        self,
        arguments: dict[str, Any],
    ) -> list[TextContent]:
        """Handle spatial audio trigger request."""
        position = arguments.get("position", 0.0)
        urgency = arguments.get("urgency", "medium")
        object_type = arguments.get("object_type", "obstacle")
        message = arguments.get("message")
        
        if self._audio_callback:
            await self._audio_callback(
                position=position,
                urgency=urgency,
                object_type=object_type,
                message=message,
            )
        
        logger.info(f"Spatial audio triggered: {object_type} at {position}, urgency={urgency}")
        
        return [TextContent(
            type="text",
            text=f"Audio cue triggered for {object_type} at position {position:.2f} with {urgency} urgency.",
        )]
    
    async def _handle_emergency_halt(
        self,
        arguments: dict[str, Any],
    ) -> list[TextContent]:
        """Handle emergency halt request."""
        reason = arguments.get("reason", "Unknown reason")
        
        self._halt_flag = True
        
        logger.warning(f"EMERGENCY HALT: {reason}")
        
        if self._audio_callback:
            await self._audio_callback(
                position=0.0,
                urgency="critical",
                object_type="halt",
                message=f"Emergency stop. {reason}",
            )
        
        return [TextContent(
            type="text",
            text=f"EMERGENCY HALT TRIGGERED: {reason}. All processing stopped.",
        )]
    
    async def _handle_ask_clarification(
        self,
        arguments: dict[str, Any],
    ) -> list[TextContent]:
        """Handle user clarification request."""
        question = arguments.get("question", "Please confirm")
        context = arguments.get("context", "")
        confidence = arguments.get("confidence", 0.0)
        detection_bbox = arguments.get("detection_bbox", [0.5, 0.5, 0.5, 0.5])
        
        center_x = (detection_bbox[0] + detection_bbox[2]) / 2 if len(detection_bbox) >= 4 else 0.5
        position = (center_x - 0.5) * 2
        
        if self._audio_callback:
            await self._audio_callback(
                position=position,
                urgency="medium",
                object_type="question",
                message=question,
            )
        
        if self._clarification_callback:
            response = await self._clarification_callback(question, context, confidence)
            return [TextContent(type="text", text=f"User response: {response}")]
        
        logger.info(f"Clarification requested: {question} (confidence: {confidence:.2%})")
        
        return [TextContent(
            type="text",
            text=f"Clarification requested: {question}. Context: {context}. Awaiting user response.",
        )]
    
    def register_audio_callback(self, callback) -> None:
        """Register callback for audio playback."""
        self._audio_callback = callback
    
    def register_clarification_callback(self, callback) -> None:
        """Register callback for user clarification responses."""
        self._clarification_callback = callback
    
    def get_resolution_mode(self) -> str:
        """Get current resolution mode."""
        return self._resolution_mode
    
    def is_halted(self) -> bool:
        """Check if system is in halt state."""
        return self._halt_flag
    
    def clear_halt(self) -> None:
        """Clear halt flag to resume processing."""
        self._halt_flag = False
        logger.info("Halt flag cleared")
    
    async def run_stdio(self) -> None:
        """Run server with stdio transport."""
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                self.server.create_initialization_options(),
            )


_mcp_server: SyntheticCortexMCPServer | None = None


def get_mcp_server() -> SyntheticCortexMCPServer:
    """Get or create the global MCP server instance."""
    global _mcp_server
    if _mcp_server is None:
        _mcp_server = SyntheticCortexMCPServer()
    return _mcp_server


async def main() -> None:
    """Entry point for MCP server."""
    server = get_mcp_server()
    await server.run_stdio()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
