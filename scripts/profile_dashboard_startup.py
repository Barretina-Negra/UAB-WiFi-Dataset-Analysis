
import subprocess
import sys
import time
import requests
import os
import signal
import re

def main():
    print("Starting Streamlit server for profiling...")
    
    # Start streamlit in background
    # We use unbuffered output (-u) and capture stderr
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["dashboard"] = "voronoi" # Set the dashboard mode as requested
    
    process = subprocess.Popen(
        ["uv", "run", "streamlit", "run", "src/integrated_dashboard.py", "--server.headless=true", "--server.port=8502"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, # Merge stderr into stdout
        env=env,
        text=True,
        bufsize=1
    )

    print("Waiting for server to be ready...")
    server_url = "http://localhost:8502"
    ready = False
    for _ in range(30): # Wait up to 30 seconds
        try:
            requests.get(server_url + "/_stcore/health")
            ready = True
            break
        except requests.exceptions.ConnectionError:
            time.sleep(1)
    
    if not ready:
        print("Server failed to start.")
        process.kill()
        return

    print("Server ready. Triggering script execution...")
    # Just accessing the main page should trigger the script in headless mode?
    # Streamlit is weird. Usually it needs a websocket connection.
    # But let's try just hitting the main page.
    try:
        requests.get(server_url)
    except Exception as e:
        print(f"Request failed: {e}")

    print("Capturing logs...")
    
    start_time = time.time()
    captured_logs = []
    
    # Read stdout line by line
    while time.time() - start_time < 60: # Wait up to 60 seconds
        line = process.stdout.readline()
        if not line:
            # If process died, break
            if process.poll() is not None:
                break
            continue
        
        # Print all lines to debug
        print(f"STDERR: {line.strip()}")

        if "[PROFILE]" in line:
            captured_logs.append(line.strip())
            
        if "Script End" in line:
            print("Script finished execution.")
            break
            
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()

    print("\n--- Profile Summary ---")
    for log in captured_logs:
        print(log)

if __name__ == "__main__":
    main()
