import os
import json

targets = ['baz_ar', 'delta_baz', 'diarrhea', 'illness_burden']

def summarize_scores():
    print("=== MAL-ED Clinical Nexus: Registry Score Report ===")
    
    for t in targets:
        print(f"\n--- Target: {t.upper()} ---")
        target_path = os.path.join('models', t)
        
        if not os.path.exists(target_path):
            print("  [ERROR] No models found for this target.")
            continue
            
        # Table Header
        print(f"{'Model':<20} | {'Metrics'}")
        print("-" * 60)
        
        for m_folder in os.listdir(target_path):
            meta_path = os.path.join(target_path, m_folder, 'metadata.json')
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    meta = json.load(f)
                    model_name = meta['model_name']
                    metrics = meta['metrics']
                    
                    # Formatting metrics string
                    metric_str = ", ".join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" for k, v in metrics.items()])
                    print(f"{model_name:<20} | {metric_str}")

if __name__ == "__main__":
    summarize_scores()
