import time
from collections import defaultdict
import numpy as np

class TimingProfiler:
    """Enhanced profiler supporting both sequential and parallel pipelines"""
    
    def __init__(self):
        self.times = defaultdict(list)
        self.start_times = {}
        
    def start(self, operation: str):
        """Start timing an operation"""
        self.start_times[operation] = time.time()
        
    def end(self, operation: str):
        """End timing an operation and store the duration"""
        if operation in self.start_times:
            duration = time.time() - self.start_times[operation]
            self.times[operation].append(duration)
            del self.start_times[operation]
            return duration
        return 0
        
    def get_summary(self):
        """Get timing summary"""
        summary = {}
        for operation, durations in self.times.items():
            summary[operation] = {
                'total': sum(durations),
                'average': np.mean(durations),
                'count': len(durations),
                'min': min(durations),
                'max': max(durations)
            }
        return summary
        
    def print_summary(self):
        """Print timing summary with automatic pipeline detection"""
        print("\n" + "="*60)
        print("📊 PIPELINE PERFORMANCE ANALYSIS")
        print("="*60)
        
        summary = self.get_summary()
        
        # Define timing categories for both sequential and parallel pipelines
        initialization_ops = [
            'config_loading', 
            'detector_initialization', 
            'pose_model_initialization'
        ]
        
        preprocessing_ops = [
            'image_loading', 
            'image_preprocessing'
        ]
        
        # Sequential pipeline operations
        sequential_detection_ops = [
            'object_detection',           # Sequential: full detection
            'detection_filtering'         # Sequential: split persons vs objects
        ]
        
        sequential_pose_ops = [
            'pose_estimation'             # Sequential: pose estimation
        ]
        
        # Parallel pipeline operations
        parallel_detection_ops = [
            'human_detection',            # Parallel: human detection first
            'parallel_processing',        # Parallel: container for parallel tasks
            'parallel_pose_estimation',   # Parallel: pose task
            'parallel_crop_detection',    # Parallel: crop-based PPE detection
            'cropping_persons',
            'batch_preparation',
            'batch_inference',
            'coordinate_transformation'
        ]
        
        tracking_ops = [
            'violation_detection'
        ]
        
        postprocessing_ops = [
            # Future: NMS, SAHI, NMM operations will go here
        ]
        
        visualization_ops = [
            'image_generation',
            'drawing_detections',
            'drawing_poses', 
            'drawing_violations',
            'drawing_pose_boxes'
        ]
        
        output_ops = [
            'saving_outputs'
        ]
        
        # Calculate category totals
        def get_category_time(ops):
            return sum(summary[op]['total'] for op in ops if op in summary)
        
        # Detect pipeline type
        is_parallel_pipeline = any(op in summary for op in parallel_detection_ops)
        
        init_time = get_category_time(initialization_ops)
        preprocessing_time = get_category_time(preprocessing_ops)
        
        if is_parallel_pipeline:
            detection_time = get_category_time(parallel_detection_ops)
            pose_time = 0  # Included in parallel processing
            pipeline_type = "PARALLEL"
        else:
            detection_time = get_category_time(sequential_detection_ops)
            pose_time = get_category_time(sequential_pose_ops)
            pipeline_type = "SEQUENTIAL"
        
        tracking_time = get_category_time(tracking_ops)
        postprocessing_time = get_category_time(postprocessing_ops)
        visualization_time = get_category_time(visualization_ops)
        output_time = get_category_time(output_ops)
        
        # Total times
        total_pipeline_time = sum(summary[op]['total'] for op in summary)
        core_time = detection_time + pose_time + tracking_time
        total_inference_time = preprocessing_time + core_time + postprocessing_time + visualization_time
        
        # Print breakdown
        print(f"🔧 INITIALIZATION ({pipeline_type} Pipeline):")
        print(f"  Model loading & setup:      {init_time:.3f}s")
        for op in initialization_ops:
            if op in summary:
                stats = summary[op]
                op_name = op.replace('_', ' ').title()
                print(f"    - {op_name:<20}: {stats['total']:.3f}s")
        
        print("\n📊 PREPROCESSING:")
        print(f"  Image preparation:          {preprocessing_time:.3f}s")
        for op in preprocessing_ops:
            if op in summary:
                stats = summary[op]
                op_name = op.replace('_', ' ').title()
                print(f"    - {op_name:<20}: {stats['total']:.3f}s ({stats['average']*1000:.1f}ms)")
        
        print("\n🎯 DETECTION & POSE:")
        print(f"  Core inference:             {core_time:.3f}s")
        
        if is_parallel_pipeline:
            # Parallel pipeline breakdown
            if 'human_detection' in summary:
                stats = summary['human_detection']
                print(f"    - Human Detection:        {stats['total']:.3f}s ({stats['average']*1000:.1f}ms)")
            
            if 'parallel_processing' in summary:
                parallel_time = summary['parallel_processing']['total']
                print(f"    - Parallel Processing:    {parallel_time:.3f}s")
                
                # Show parallel breakdown
                pose_time_parallel = summary.get('parallel_pose_estimation', {}).get('total', 0)
                crop_time = summary.get('parallel_crop_detection', {}).get('total', 0)
                
                if pose_time_parallel > 0:
                    print(f"      • Pose Estimation:      {pose_time_parallel:.3f}s ({pose_time_parallel*1000:.1f}ms)")
                if crop_time > 0:
                    print(f"      • Crop PPE Detection:   {crop_time:.3f}s ({crop_time*1000:.1f}ms)")
                
                # Parallel efficiency
                max_parallel = max(pose_time_parallel, crop_time)
                if max_parallel > 0:
                    efficiency = max_parallel / parallel_time if parallel_time > 0 else 0
                    print(f"      • Parallel Efficiency:  {efficiency:.1%}")
        else:
            # Sequential pipeline breakdown
            if 'object_detection' in summary:
                stats = summary['object_detection']
                print(f"    - Object Detection:       {stats['total']:.3f}s ({stats['average']*1000:.1f}ms)")
            
            if 'detection_filtering' in summary:
                stats = summary['detection_filtering']
                print(f"    - Detection Filtering:    {stats['total']:.3f}s ({stats['average']*1000:.1f}ms)")
            
            if 'pose_estimation' in summary:
                stats = summary['pose_estimation']
                print(f"    - Pose Estimation:        {stats['total']:.3f}s ({stats['average']*1000:.1f}ms)")
        
        print("\n🔍 TRACKING:")
        print(f"  Violation detection:        {tracking_time:.3f}s")
        if 'violation_detection' in summary:
            stats = summary['violation_detection']
            print(f"    - Violation Analysis:     {stats['total']:.3f}s ({stats['average']*1000:.1f}ms)")
        
        if postprocessing_time > 0:
            print(f"\n⚙️  POSTPROCESSING:")
            print(f"  Additional processing:      {postprocessing_time:.3f}s")
        
        print(f"\n🎨 VISUALIZATION:")
        print(f"  Drawing & rendering:        {visualization_time:.3f}s")
        for op in visualization_ops:
            if op in summary:
                stats = summary[op]
                if stats['total'] > 0.001:  # Only show significant times
                    op_name = op.replace('_', ' ').title()
                    print(f"    - {op_name:<20}: {stats['total']:.3f}s ({stats['average']*1000:.1f}ms)")
        
        print(f"\n💾 OUTPUT:")
        print(f"  Saving results:             {output_time:.3f}s")
        
        print("\n" + "-"*60)
        print("📈 PERFORMANCE SUMMARY:")
        print("-"*60)
        print(f"🔧 Initialization time:       {init_time:.3f}s")
        print(f"🚀 Total inference time:      {total_inference_time:.3f}s")
        print(f"🎯 Core inference time:       {core_time:.3f}s")
        print(f"📊 Total pipeline time:       {total_pipeline_time:.3f}s")
        
        print("\n" + "="*50)
        print("📊 FPS ANALYSIS")
        print("="*50)
        
        # Calculate proper FPS
        if core_time > 0:
            core_fps = 1.0 / core_time
            print(f"🎯 Core Inference FPS:        {core_fps:.2f} fps")
            print(f"   (Detection + Pose + Track)  ({core_time:.3f}s)")
        
        if total_inference_time > 0:
            total_fps = 1.0 / total_inference_time
            print(f"🚀 Total Inference FPS:       {total_fps:.2f} fps")
            print(f"   (Including visualization)   ({total_inference_time:.3f}s)")
        
        print(f"📊 Initialization overhead:   {init_time:.3f}s")
        print("="*50)
        
        # # Research paper metrics
        # if is_parallel_pipeline:
        #     detection_time_ms = summary.get('human_detection', {}).get('average', 0) * 1000
        #     pose_time_ms = summary.get('parallel_pose_estimation', {}).get('average', 0) * 1000
        # else:
        #     detection_time_ms = summary.get('object_detection', {}).get('average', 0) * 1000
        #     pose_time_ms = summary.get('pose_estimation', {}).get('average', 0) * 1000
        
        # tracking_time_ms = summary.get('violation_detection', {}).get('average', 0) * 1000
        
        # print(f"\n💡 For research papers:")
        # print(f"   • Core inference speed: {core_fps:.1f} fps")
        # print(f"   • Detection time: {detection_time_ms:.1f}ms")
        # print(f"   • Pose estimation time: {pose_time_ms:.1f}ms")
        # print(f"   • Tracking time: {tracking_time_ms:.1f}ms")
        # print(f"   • Pipeline type: {pipeline_type}")
        # print("="*60)