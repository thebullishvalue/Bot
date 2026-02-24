import subprocess
import sys
import os

def main():
    script_name = "run.sh"
    
    # Check if the shell script actually exists in the current directory
    if not os.path.exists(script_name):
        print(f"Error: '{script_name}' not found in the current directory.")
        sys.exit(1)

    try:
        # Step 1: chmod +x run.sh
        print(f"Setting executable permissions on {script_name}...")
        subprocess.run(["chmod", "+x", script_name], check=True)

        # Step 2: ./run.sh both
        print(f"Executing ./{script_name} both...\n" + "-" * 40)
        
        # By not capturing the output, the stdout and stderr will stream 
        # directly to your terminal just like running it normally.
        subprocess.run([f"./{script_name}", "both"], check=True)
        
        print("-" * 40 + "\nExecution completed successfully.")

    except subprocess.CalledProcessError as e:
        # This triggers if either the chmod or the run.sh command fails
        print(f"\nError: The process failed with exit code {e.returncode}.")
        sys.exit(e.returncode)
        
    except Exception as e:
        # Catch-all for any other unexpected Python errors
        print(f"\nAn unexpected error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
