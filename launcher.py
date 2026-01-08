import os
import sys
import argparse
import subprocess
import signal
import time

# List to keep track of spawned processes
processes = []

def signal_handler(sig, frame):
    """
    Handle Ctrl+C (SIGINT) to cleanly kill all child processes.
    """
    print("\n[Launcher] Received interrupt signal. Terminating processes...")
    for p in processes:
        if p.poll() is None:  # If process is still running
            p.terminate()
    sys.exit(0)

def main():
    parser = argparse.ArgumentParser(description="Custom Torchrun-like Launcher for C++/CUDA")
    
    # Distributed args
    parser.add_argument("--nproc_per_node", type=int, required=True, 
                        help="Number of processes (GPUs) to launch on this node.")
    parser.add_argument("--nnodes", type=int, default=1, 
                        help="Total number of nodes in the cluster.")
    parser.add_argument("--node_rank", type=int, default=0, 
                        help="Rank of this specific node (0 for single node, 0 to N-1 for multi-node).")
    parser.add_argument("--master_addr", type=str, default="127.0.0.1", 
                        help="IP address of the node 0 (Master).")
    parser.add_argument("--master_port", type=str, default="29500", 
                        help="Port on Master for communication.")
    
    # User script/binary args
    parser.add_argument("training_binary", type=str, help="Path to your C++ executable.")
    parser.add_argument("training_args", nargs=argparse.REMAINDER, 
                        help="Arguments to pass to the C++ executable.")

    args = parser.parse_args()

    # Register signal handler for cleanup
    signal.signal(signal.SIGINT, signal_handler)

    # Calculate Global World Size
    world_size = args.nproc_per_node * args.nnodes
    
    print(f"[Launcher] Starting distributed run")
    print(f" -> Master: {args.master_addr}:{args.master_port}")
    print(f" -> World Size: {world_size}")
    print(f" -> Processes per node: {args.nproc_per_node}")
    print(f" -> Node Rank: {args.node_rank}")
    print("---------------------------------------------------------------")

    for local_rank in range(args.nproc_per_node):
        # Calculate Global Rank
        # Formula: (Node_Rank * Procs_Per_Node) + Local_Rank
        global_rank = (args.node_rank * args.nproc_per_node) + local_rank

        # Prepare Environment Variables for the child process
        current_env = os.environ.copy()
        
        # 1. Distributed Context
        current_env["RANK"] = str(global_rank)
        current_env["LOCAL_RANK"] = str(local_rank)
        current_env["WORLD_SIZE"] = str(world_size)
        current_env["LOCAL_WORLD_SIZE"] = str(args.nproc_per_node)
        
        # 2. Network Info
        current_env["MASTER_ADDR"] = args.master_addr
        current_env["MASTER_PORT"] = args.master_port

        # 3. GPU Isolation (Mimic torchrun behavior)
        # We map Local Rank 0 -> GPU 0, Local Rank 1 -> GPU 1, etc.
        # This simplifies your C++ code: it can always just select device 0!
        current_env["CUDA_VISIBLE_DEVICES"] = str(local_rank)

        # Build the command
        cmd = [args.training_binary] + args.training_args

        # Spawn the process
        print(f"[Launcher] Spawning Process: Rank {global_rank} (Local {local_rank}) mapped to GPU {local_rank}")
        
        # Start process async
        p = subprocess.Popen(cmd, env=current_env)
        processes.append(p)

    # Monitor loop: Wait for all processes to finish
    try:
        while True:
            all_finished = True
            for p in processes:
                if p.poll() is None:
                    all_finished = False
                    break
                else:
                    # If any process exits with error, kill all others (Gang semantics)
                    if p.returncode != 0:
                        print(f"[Launcher] Process exited with error code {p.returncode}. Killing others.")
                        for kp in processes:
                            if kp.poll() is None: kp.terminate()
                        sys.exit(p.returncode)
            
            if all_finished:
                print("[Launcher] All processes completed successfully.")
                break
            
            time.sleep(0.5)
            
    except Exception as e:
        print(f"[Launcher] Error occurred: {e}")
        signal_handler(None, None)

if __name__ == "__main__":
    main()


#  python3 launcher.py --nproc_per_node=2 ./launcher