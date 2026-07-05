import sys
INTENT_RESPONSE_MAP = {
    "hello": "Greetings! Welcome to DecodeLabs. How can I assist you with your AI training today?",
    "hi": "Hello! Professional AI Assistant ready. Please state your query.",
    "hey": "Hi there! How can I guide your programmatic systems today?",
    "help": "Available options:\n - Ask for system status by typing 'status'\n - Learn about the developer by typing 'credits'\n - Exit the loop by typing 'exit' or 'quit'",
    "status": "System Status: Online. All rule-based validation guardrails are active and stable.",
    "credits": "Engineered by Saad Afridi",
}

EXIT_COMMANDS = {"exit", "quit", "goodbye", "bye"}

def sanitize_input(raw_text: str) -> str:
    """
    PHASE 1: INPUT & SANITIZATION
    Normalizes text data inflow by eliminating structural anomalies like 
    irregular casing and accidental leading/trailing whitespaces.
    """
    if not raw_text:
        return ""
    # Standardizing incoming stream data for predictable matching
    return raw_text.lower().strip()


def evaluate_intent(clean_text: str) -> tuple[str, bool]:
    """
    PHASE 2: PROCESS (The Logic Skeleton)
    Applies strict deterministic matching logic across sanitized input 
    to map user intent directly to explicit outputs or trigger exit flags.
    """
    # Check for terminal state execution
    if clean_text in EXIT_COMMANDS:
        exit_feedback = "Terminating interface connection safely. Goodbye!"
        return exit_feedback, True
        
    # Standard deterministic dictionary fallback lookup
    response = INTENT_RESPONSE_MAP.get(
        clean_text, 
        "Error: Standard Command Not Found. Type 'help' to review valid system options."
    )
    return response, False


def execute_chatbot_pipeline() -> None:
    """
    PHASE 3: OUTPUT & SYSTEM CONTINUOUS LOOP
    Main runtime environment loop maintaining application life cycle state
    and processing input streams securely.
    """
    print("=" * 60)
    print("AI Bot is Presented To")
    print("Status: Active Guardrails Enabled. Type 'exit' to terminate.")
    print("=" * 60)
    
    while True:
        try:
            # Capturing the raw user feed interaction
            raw_input_stream = input("\nYou: ")
            
            # Step 1: Pass through the Sanitization Filter
            processed_input = sanitize_input(raw_input_stream)
            
            # Drop cycles on completely blank/empty inputs
            if not processed_input:
                continue
                
            # Step 2 & 3: Process Logic Matrix & Evaluate State Outputs
            system_response, terminal_signal_triggered = evaluate_intent(processed_input)
            
            # Output dispatching sequence
            print(f"Bot: {system_response}")
            
            # Graceful, controlled thread shutdown
            if terminal_signal_triggered:
                print("=" * 60)
                sys.exit(0)
                
        except (KeyboardInterrupt, SystemExit):
            print("\n\n[ALERT] System forced offline safely via interrupt vector. Goodbye.")
            break
        except Exception as e:
            print(f"\n[CRITICAL ERROR] Pipeline execution failure: {str(e)}")

if __name__ == "__main__":
    execute_chatbot_pipeline()