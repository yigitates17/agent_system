# checkpoints/user_approval.py

from core.schemas import CheckpointResponse, CheckpointDecision, ToolResult
from checkpoints.base_checkpoint import BaseCheckpointHandler


class ConsoleApprovalHandler(BaseCheckpointHandler):
    """Interactive console-based approval."""
    
    async def handle(self, step_name: str, result: ToolResult) -> CheckpointResponse:
        print(f"\n{'='*50}")
        print(f"CHECKPOINT: {step_name}")
        print(f"{'='*50}")
        
        if result.success:
            print(f"Result: {result.data}")
        else:
            print(f"Error: {result.error}")
        
        print("\nOptions:")
        print("  [a] Approve - continue to next step")
        print("  [r] Revise  - retry this step with feedback")
        print("  [b] Back    - go back to a previous step")
        print("  [s] Stop    - stop workflow")
        
        choice = input("\nChoice: ").strip().lower()
        
        if choice == "a":
            return CheckpointResponse(decision=CheckpointDecision.APPROVE)
        
        elif choice == "r":
            feedback = input("Feedback for revision: ")
            return CheckpointResponse(
                decision=CheckpointDecision.REVISE,
                feedback=feedback,
            )
        
        elif choice == "b":
            step = input("Go back to step: ")
            return CheckpointResponse(
                decision=CheckpointDecision.GO_BACK,
                go_back_to=step,
            )
        
        else:
            return CheckpointResponse(decision=CheckpointDecision.STOP)


class AutoApproveHandler(BaseCheckpointHandler):
    """Auto-approves everything. Useful for testing."""
    
    async def handle(self, step_name: str, result: ToolResult) -> CheckpointResponse:
        return CheckpointResponse(decision=CheckpointDecision.APPROVE)


class CallbackApprovalHandler(BaseCheckpointHandler):
    """For web/API usage - delegates to a callback."""
    
    def __init__(self, callback):
        self.callback = callback
    
    async def handle(self, step_name: str, result: ToolResult) -> CheckpointResponse:
        return await self.callback(step_name, result)