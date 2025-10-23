import time

def profile_pipeline(cap, engine, detector, face_mesh, num_frames=30):
    """Profile each component to find bottlenecks."""
    
    print("\n" + "="*80)
    print("PERFORMANCE PROFILING")
    print("="*80)
    
    timings = {
        'frame_read': [],
        'face_detection': [],
        'gaze_estimation': [],
        'face_mesh': [],
        'total': []
    }
    
    for i in range(num_frames):
        frame_start = time.time()
        
        # 1. Frame reading
        t0 = time.time()
        ret, frame = cap.read()
        if not ret:
            break
        timings['frame_read'].append(time.time() - t0)
        
        # 2. Face detection
        t0 = time.time()
        bboxes, _ = detector.detect(frame)
        timings['face_detection'].append(time.time() - t0)
        
        if len(bboxes) > 0:
            # 3. Gaze estimation
            x_min, y_min, x_max, y_max = map(int, bboxes[0][:4])
            face_img = frame[y_min:y_max, x_min:x_max]
            
            if face_img.size > 0:
                t0 = time.time()
                pitch, yaw = engine.estimate(face_img)
                timings['gaze_estimation'].append(time.time() - t0)
        
        # 4. Face mesh (for blink detection)
        t0 = time.time()
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)
        timings['face_mesh'].append(time.time() - t0)
        
        timings['total'].append(time.time() - frame_start)
    
    # Print results
    print(f"\nTested {num_frames} frames:")
    print("-" * 80)
    
    for component, times in timings.items():
        if times:
            avg_time = np.mean(times) * 1000  # Convert to ms
            max_time = np.max(times) * 1000
            fps = 1.0 / np.mean(times) if np.mean(times) > 0 else 0
            print(f"{component:20s}: {avg_time:6.1f}ms avg, {max_time:6.1f}ms max, {fps:5.1f} FPS")
    
    print("\n" + "="*80)
    print("BOTTLENECK ANALYSIS:")
    print("="*80)
    
    # Find the slowest component
    avg_times = {k: np.mean(v) * 1000 for k, v in timings.items() if v and k != 'total'}
    slowest = max(avg_times, key=avg_times.get)
    print(f"\nSlowest component: {slowest} ({avg_times[slowest]:.1f}ms)")
    
    # Calculate percentages
    total_avg = np.mean(timings['total']) * 1000
    print(f"\nTime breakdown:")
    for component, times in timings.items():
        if times and component != 'total':
            pct = (np.mean(times) * 1000 / total_avg) * 100
            print(f"  {component:20s}: {pct:5.1f}%")