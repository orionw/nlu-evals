import os
import pandas as pd
import glob

def process_glue_results(filepath):
    """Process GLUE results from a CSV file."""
    try:
        with open(filepath, 'r') as f:
            line = f.readlines()[1]
            if not line:
                return None
            
            values = line.split(',')
            if len(values) < 9:  # Model name + 8 metrics
                return None
                
            return {
                'model': values[0],
                'avg glue': float(values[1]),
                'cola': float(values[2]),
                'mnli': float(values[3]),
                'mrpc': float(values[4]),
                'qnli': float(values[5]),
                'qqp': float(values[6]),
                'rte': float(values[7]),
                'sst2': float(values[8]),
                'stsb': float(values[9]),
            }
    except:
        return None

def process_xtreme_results(filepath):
    """Process XTREME results from a CSV file."""
    try:
        # Read just the first line to get the headers
        with open(filepath, 'r') as f:
            header_line = f.readline().strip()
            data_line = f.readline().strip()
            
        if not header_line or not data_line:
            return None
            
        headers = header_line.split(',')
        values = data_line.split(',')
        
        if len(headers) != len(values):
            return None
            
        # Create a dictionary of header:value pairs
        data_dict = dict(zip(headers, values))
        
        return {
            'model': data_dict['model'],
            'xnli': float(data_dict['xnli-predict_accuracy']),
            'paws-x': float(data_dict['paws-x-predict_accuracy']),
            # NOTE: can't run QA yet until the PR gets merged
            # 'xquad': float(data_dict['xquad-test_f1']),
            # 'mlqa': float(data_dict['mlqa-test_f1']),
            # 'tydiqa': float(data_dict['tydiqa-test_f1']),
            'xcopa': float(data_dict['xcopa-predict_accuracy']),
            'mewslix': float(data_dict['mewslix-map_at_20']),
            "wikiann": float(data_dict['wikiann-predict_f1']),
            "udpos": float(data_dict['udpos-predict_f1']),
        }
    except Exception as e:
        print(f"Error processing XTREME results: {e}")
        return None

def main():
    # Get all immediate subdirectories in results/
    subdirs = [d for d in glob.glob('results/*/') if os.path.isdir(d)]
    
    all_results = []
    
    for subdir in subdirs:
        model_results = {}
        
        # Try to get GLUE results
        glue_path = os.path.join(subdir, 'glue.csv')
        if os.path.exists(glue_path):
            glue_results = process_glue_results(glue_path)
            if glue_results:
                model_results = glue_results
        
        # Try to get XTREME results
        xtreme_path = os.path.join(subdir, 'xtreme.csv')
        if os.path.exists(xtreme_path):
            xtreme_results = process_xtreme_results(xtreme_path)
            if xtreme_results:
                # If we already have results for this model, update them
                if model_results:
                    model_results.update({k:v for k,v in xtreme_results.items() if k != 'model'})
                else:
                    model_results = xtreme_results
        
        if model_results:
            all_results.append(model_results)
    
    # Convert to DataFrame
    if all_results:
        df = pd.DataFrame(all_results)
        # Set model as index
        df.set_index('model', inplace=True)
        # Save to CSV
        df.to_csv('combined_results.csv')
        print(f"Results saved to combined_results.csv")
        print("\nColumns in output:")
        print(df.columns.tolist())
        print("\nFirst few rows:")
        print(df.head())
    else:
        print("No results found!")

if __name__ == "__main__":
    main()