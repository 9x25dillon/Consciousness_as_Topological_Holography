#!/usr/bin/env python3
"""
run_demos.py
Simple runner that executes all demos from the topological consciousness framework
"""

from topological_consciousness import run_all

if __name__ == "__main__":
    print("="*70)
    print("TOPOLOGICAL HOLOGRAPHY: CONSCIOUSNESS AS (2+1)D TQFT")
    print("="*70)
    print()
    
    run_all(seed=0)
    
    print("\n" + "="*70)
    print("All demos completed. See rg_flow.png for the generated plot.")
    print("="*70)