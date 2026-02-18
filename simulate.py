from muscle import Muscle
from skeleton import Skeleton
from env import Simulation
import numpy as np
import matplotlib
#matplotlib.use('Agg')  # Use non-interactive backend for headless saving
import matplotlib.pyplot as plt
import os
from datetime import datetime


def get_max_height(results):
    """
    Calculate and print the maximum hopping height.
    """
    max_height = np.max(results['height'])
    #print(f"Maximum hopping height: {max_height:.4f} m")
    return max_height


def find_stance_cycles(results):
    """
    Find all stance phase cycles in the simulation.
    """
    phase = results['phase']
    cycles = []
    in_stance = False
    start_idx = 0
    
    for i in range(len(phase)):
        if phase[i] and not in_stance:
            # Entering stance
            start_idx = i
            in_stance = True
        elif not phase[i] and in_stance:
            # Exiting stance
            cycles.append((start_idx, i))
            in_stance = False
    
    return cycles


def plot_single_stance_cycle(results, cycle_num=3, save_dir=None):
    """
    Plot a single stance cycle with force, activation, and foot height.
    """
    cycles = find_stance_cycles(results)
    
    if cycle_num > len(cycles):
        print(f"Warning: Only {len(cycles)} stance cycles found. Plotting last cycle.")
        cycle_idx = len(cycles) - 1
    else:
        cycle_idx = cycle_num - 1
    
    start_idx, end_idx = cycles[cycle_idx]
    
    # Extract data for this cycle
    time_cycle = results['time'][start_idx:end_idx]
    time_cycle = (time_cycle - time_cycle[0]) * 1000  
    
    f_leg = results['leg_force'][start_idx:end_idx]
    act = results['activation'][start_idx:end_idx]
    foot_height = results['height'][start_idx:end_idx] - results['leg_length'][start_idx:end_idx]
    
    # Create figure with multiple y-axes
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Plot force on left y-axis
    color1 = 'tab:blue'
    ax1.set_xlabel('Time (ms)', fontsize=12)
    ax1.set_ylabel('Leg Force (N)', color=color1, fontsize=12)
    line1 = ax1.plot(time_cycle, f_leg, color=color1, linewidth=2.5, label='Leg Force')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, alpha=0.3)
    
    # Create second y-axis for activation
    ax2 = ax1.twinx()
    color2 = 'tab:orange'
    ax2.set_ylabel('Activation', color=color2, fontsize=12)
    line2 = ax2.plot(time_cycle, act, color=color2, linewidth=2.5, linestyle='--', label='Activation')
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_ylim([0, 1.1])
    
    # Add vertical lines at start and end of stance 
    ax1.axvline(time_cycle[0], color='gray', linestyle=':', linewidth=2, alpha=0.5, label='Stance Start')
    ax1.axvline(time_cycle[-1], color='gray', linestyle=':', linewidth=2, alpha=0.5, label='Stance End')
    
    # Title and legend
    fig.suptitle(f'Single Stance Cycle #{cycle_idx + 1}', fontsize=14, fontweight='bold')
    
    # Combine legends from all axes
    lines = line1 + line2 
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left', fontsize=10)
    
    fig.tight_layout()
    
    if save_dir:
        save_path = os.path.join(save_dir, 'stance_cycle_analysis.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved stance cycle plot to {save_path}")
        plt.close(fig) 
    else:
        plt.show()
        plt.close(fig)
    
def create_results_folder():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_root = os.path.dirname(script_dir)
    
    results_dir = os.path.join(workspace_root, 'results')
    
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # Create timestamped subfolder
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(results_dir, timestamp)
    
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)
    
    print(f"Results will be saved to: {run_dir}")
    return run_dir


def main():
    """
    Main simulation loop for hopping model.
    """
    run_dir = create_results_folder()
    
    # Simulation parameters
    dt = 0.00005  # timestep (s)
    t_max = 10  # total simulation time (s)
    delay_time = 15 #in milliseconds, based on paper
    delay_steps = int(delay_time / (dt * 1000))  # feedback delay in timesteps
    show_animation = True  # Set to True to see animation, False for plots only
    save_video = False  # If True, save mp4 only; if False, show animation window
    
    # Muscle parameters
    F_max_iso = 22000  # Maximum isometric force (N)
    l_opt = 0.1        # Optimal fiber length (m)
    l_slack = 0.4      # Tendon slack length (m)
    v_max = 12 * l_opt # Maximum contraction velocity (m/s)
    N = 1.5            # Force-velocity shape parameter
    K = 5              # Force-velocity shape parameter
    l_ref = 0.5        # Reference muscle length (m)
    
    # Initialize muscle
    muscle = Muscle(
        name="muscle",
        F_max_iso=F_max_iso,
        l_opt=l_opt,
        l_slack=l_slack,
        v_max=v_max,
        N=N,
        K=K,
        l_ref=l_ref,
        dt=dt,
        eps_ref=0.04,
        max_force_history=1000  # Store last 1000 timesteps for delayed feedback
    )
    
    # Initialize skeleton
    skeleton = Skeleton(muscle)
    
    # Initialize simulation with animation option
    sim = Simulation(
        muscle,
        skeleton,
        dt=dt,
        t_max=t_max,
        animate=show_animation,
        save_dir=run_dir,
        save_video=save_video,
    )
    
    # Run simulation
    print(f"Running simulation for {t_max}s with dt={dt}s...")
    print(f"Total timesteps: {len(sim.time)}")
    print(f"Feedback delay: {delay_steps} steps ({delay_steps*dt:.6f}s)")
    print(f"Animation enabled: {show_animation}")
    
    results = sim.run(delay_steps=delay_steps)
    
    print("Simulation complete!")
    print()
    
    # Get max height
    max_h = get_max_height(results)
    print()
    
    # Plot full results
    plot_results(results, save_dir=run_dir)
    
    # Plot a single stance cycle (2nd cycle to avoid transient)
    plot_single_stance_cycle(results, cycle_num=6, save_dir=run_dir)


def plot_results(results, save_dir=None):
    """
    Plot simulation results.
    """
    fig, axes = plt.subplots(4, 2, figsize=(14, 10))
    fig.suptitle('Hopping Model Simulation Results', fontsize=14, fontweight='bold')
    
    time = results['time']
    
    # Height
    axes[0, 0].plot(time, results['height'], 'b-', linewidth=1.5)
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Height (m)')
    axes[0, 0].set_title('Center of Mass Height')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Velocity
    axes[0, 1].plot(time, results['velocity'], 'r-', linewidth=1.5)
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Velocity (m/s)')
    axes[0, 1].set_title('Vertical Velocity')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Muscle force
    axes[1, 0].plot(time, results['muscle_force'], 'g-', linewidth=1.5)
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Force (N)')
    axes[1, 0].set_title('Muscle Force')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Leg force
    axes[1, 1].plot(time, results['leg_force'], 'm-', linewidth=1.5)
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Force (N)')
    axes[1, 1].set_title('Leg Force (Ground Reaction)')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Activation
    axes[2, 0].plot(time, results['activation'], 'c-', linewidth=1.5)
    axes[2, 0].set_xlabel('Time (s)')
    axes[2, 0].set_ylabel('Activation')
    axes[2, 0].set_title('Muscle Activation')
    axes[2, 0].set_ylim([0, max(results['activation'])*1.2])
    axes[2, 0].grid(True, alpha=0.3)
    
    # Stimulation
    axes[2, 1].plot(time, results['stimulation'], 'orange', linewidth=1.5)
    axes[2, 1].set_xlabel('Time (s)')
    axes[2, 1].set_ylabel('Stimulation')
    axes[2, 1].set_title('Muscle Stimulation (with Feedback)')
    axes[2, 1].grid(True, alpha=0.3)

    #plot foot height
    foot_height = results['height'] - results['leg_length']  # foot is at height - leg length
    axes[3, 0].plot(time, foot_height, 'brown', linewidth=1.5)
    axes[3, 0].set_xlabel('Time (s)')
    axes[3, 0].set_ylabel('Foot Height (m)')
    axes[3, 0].set_title('Foot Height (Ground Contact)')
    axes[3, 0].grid(True, alpha=0.3)

    #plot phase vs time
    axes[3, 1].plot(time, results['phase'], 'purple', linewidth=1.5)
    axes[3, 1].set_xlabel('Time (s)')
    axes[3, 1].set_ylabel('Phase (0=Flight, 1=Stance)')
    axes[3, 1].set_title('Hopping Phase (Stance vs Flight)')
    axes[3, 1].set_ylim([-0.2, 1.2])
    axes[3, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_dir:
        save_path = os.path.join(save_dir, 'simulation_results.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved simulation results plot to {save_path}")
        plt.close(fig)  # Close figure after saving
    else:
        plt.show()
        plt.close(fig)


if __name__ == "__main__":
    main()
