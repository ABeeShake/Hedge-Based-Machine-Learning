"""ExpMethods/timing.py — Lightweight wall-clock profiling utilities.

Provides a context-manager ``Timer`` and a singleton ``TimingRegistry`` that
accumulate per-category execution times across an entire simulation run and
report summary statistics (count, total, mean, std) at the end.

Usage example
-------------
    from ExpMethods.timing import Timer, TimingRegistry

    with Timer("Expert Training: NODE"):
        trainer.fit(model, data_module)

    TimingRegistry.print_stats()   # prints a formatted table at run end

The results of ``TimingRegistry.get_timings()`` are also consumed by
``parse_timings.py`` and ``summarize_timings.py`` to produce the computational
timing table reported in the paper (Table: Computational Timing Analysis).
"""
import time

from collections import defaultdict
import statistics

class TimingRegistry:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(TimingRegistry, cls).__new__(cls)
            cls._instance.timings = defaultdict(list)
        return cls._instance
    
    @classmethod
    def reset(cls):
        if cls._instance:
            cls._instance.timings = defaultdict(list)
            
    @classmethod
    def get_timings(cls):
        if cls._instance is None:
            return {}
        return dict(cls._instance.timings)

    @classmethod
    def print_stats(cls):
        if cls._instance is None or not cls._instance.timings:
            print("No timings recorded.")
            return

        print("\n=== Timing Analysis Report ===")
        print(f"{'Category':<40} | {'Count':<8} | {'Total (s)':<12} | {'Mean (s)':<12} | {'Std Dev (s)':<12}")
        print("-" * 100)
        
        for category, times in sorted(cls._instance.timings.items()):
            count = len(times)
            total = sum(times)
            mean = statistics.mean(times)
            std = statistics.stdev(times) if count > 1 else 0.0
            
            print(f"{category:<40} | {count:<8} | {total:<12.4f} | {mean:<12.4f} | {std:<12.4f}")
        print("=" * 100 + "\n")


class Timer:
    def __init__(self, category):
        self.category = category
        self.registry = TimingRegistry()
        self.start_time = None
        
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.perf_counter()
        duration = end_time - self.start_time
        self.registry.timings[self.category].append(duration)
