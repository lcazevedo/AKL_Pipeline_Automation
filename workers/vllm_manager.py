import yaml
import subprocess
import os

class VLLMManager:
    def __init__(self, config_path="pipeline_config.yaml"):
        self.config_path = config_path
        self.container_name = "akl_vllm"
        self.config = self._load_config()

    def _load_config(self):
        """Reads the YAML file and returns the configuration dictionary."""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)

    def start_server(self):
        """Builds and executes the docker run command based on the YAML configuration."""
        print("Reading vLLM configurations...")
        
        try:
            # Fetch the model name from the general models section
            model_name = self.config['models']['llm_extraction']
            # Fetch server-specific configurations
            server_cfg = self.config['step7_vllm']['server']
        except KeyError as e:
            raise KeyError(f"Missing required configuration section in YAML: {e}") from e

        # Resolve the cache directory path to map into the container
        cache_dir = os.path.expanduser("~/.cache/huggingface")
        os.makedirs(cache_dir, exist_ok=True)

        # Assemble the Docker command
        cmd = [
            "docker", "run", "-d", "--rm",
            "--name", self.container_name,
            "--runtime", "nvidia",
            "--ipc", "host",
            "-p", f"{server_cfg.get('port', 8000)}:8000",
            "-v", f"{cache_dir}:/root/.cache/huggingface",
            "vllm/vllm-openai:latest",
            "--model", str(model_name),
            "--max-model-len", str(server_cfg['max_model_len']),
            "--dtype", str(server_cfg['dtype']),
            "--gpu-memory-utilization", str(server_cfg['gpu_memory_utilization'])
        ]

        # Reproducibility: server-side default seed (per-request 'seed' still applies)
        seed = self.config.get('reproducibility', {}).get('seed', server_cfg.get('seed', 42))
        cmd += ["--seed", str(int(seed))]

        if server_cfg.get('enforce_eager', False):
            cmd.append("--enforce-eager")

        print(f"Starting container {self.container_name} with model {model_name}...")
        
        try:
            # Check if a container with this name is already running and forcefully remove it
            subprocess.run(["docker", "rm", "-f", self.container_name], 
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            # Execute the run command WITHOUT capture_output so Docker pull progress is visible
            subprocess.run(cmd, check=True)
            print("✅ Container instantiated. Attaching to logs to monitor startup progress...")
            
            # Call the log monitor to show the model loading steps
            self._monitor_startup()
            
        except subprocess.CalledProcessError as e:
            print("❌ Error starting the container. Check if Docker is running and NVIDIA runtime is configured.")

    def _monitor_startup(self):
        """Streams container logs to the console until the server is fully ready."""
        # Use Popen to stream logs in real-time
        log_cmd = ["docker", "logs", "-f", self.container_name]
        process = subprocess.Popen(log_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        
        print("-" * 60)
        # Read the output line by line as it is generated
        for line in process.stdout:
            clean_line = line.strip()
            if clean_line:
                print(f"[vLLM] {clean_line}")
            
            # Check for success messages indicating the server is online
            if "Uvicorn running on" in clean_line or "Application startup complete" in clean_line:
                print("-" * 60)
                print("✅ vLLM server is online and ready to receive requests!")
                process.terminate()  # Stop trailing the logs, release the terminal
                return
        
        # If the loop finishes without returning, the container likely crashed
        process.wait()
        if process.returncode != 0:
            print("❌ Container stopped unexpectedly. Check the logs above for OOM (Out of Memory) or other errors.")

    def stop_server(self):
        """Stops and removes the vLLM container."""
        print(f"Stopping container {self.container_name}...")
        subprocess.run(["docker", "stop", self.container_name], 
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("✅ Container stopped.")

# Example usage for your orchestrator script
if __name__ == "__main__":
    manager = VLLMManager("pipeline_config.yaml")
    
    # Start the server
    manager.start_server()
    
    # At this point, your orchestrator would continue with Step 7 logic...
    
    # When the pipeline is finished (or within a try/finally block):
    #manager.stop_server()