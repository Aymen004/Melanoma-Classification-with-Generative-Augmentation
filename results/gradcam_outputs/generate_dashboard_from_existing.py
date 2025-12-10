"""
Generate HTML dashboard from existing Grad-CAM heatmaps
"""

from pathlib import Path
import json

def build_html_dashboard(output_dir):
    """Build HTML dashboard from existing heatmaps"""
    
    # Scan all existing model directories
    model_dirs = sorted([d for d in Path(output_dir).iterdir() if d.is_dir()])
    
    if not model_dirs:
        print("No model directories found!")
        return
    
    # Collect all heatmap info
    heatmap_data = []
    
    # Known architecture names
    ARCHITECTURES = ['ResNet18', 'DenseNet', 'ViT', 'Swin', 'BioViT', 'MedViT']
    
    for model_dir in model_dirs:
        model_name = model_dir.name
        
        # Parse architecture and dataset correctly
        # Handle multi-part names like "ResNet18_DCGAN_UPSCALED"
        architecture = None
        dataset = None
        
        for arch in ARCHITECTURES:
            if model_name.startswith(arch + '_'):
                architecture = arch
                dataset = model_name[len(arch)+1:]  # Everything after "Architecture_"
                break
        
        if architecture is None:
            # Fallback for unknown architectures
            parts = model_name.split('_', 1)
            architecture = parts[0]
            dataset = parts[1] if len(parts) > 1 else "UNKNOWN"
        
        # Find all heatmap images
        for img_path in sorted(model_dir.glob("*.jpg")):
            filename = img_path.name
            
            # Parse filename: SOURCE_imagename.jpg
            if '_' in filename:
                source_dataset = filename.split('_')[0]
                
                # Map source to match training dataset naming convention
                # Files are named with base dataset (DCGAN, DDPM, ORIGINAL)
                # but we want to show which actual training set was used
                source_display = source_dataset
                if 'UPSCALED' in dataset:
                    # Model was trained on upscaled data
                    if source_dataset in ['DCGAN', 'DDPM']:
                        source_display = f"{source_dataset}_UPSCALED"
                
                # Determine label from directory structure or filename
                label = "Unknown"
                if 'benign' in filename.lower():
                    label = "Benign"
                elif 'malignant' in filename.lower() or 'malware' in filename.lower():
                    label = "Malignant"
                else:
                    # Assume first 25 are benign, last 25 are malignant
                    img_index = int(''.join(filter(str.isdigit, filename[:10])) or 0)
                    label = "Benign" if img_index < 25 else "Malignant"
                
                heatmap_data.append({
                    'model': model_name,
                    'architecture': architecture,
                    'dataset': dataset,
                    'source': source_display,
                    'label': label,
                    'path': f"{model_name}/{filename}",
                    'filename': filename
                })
    
    print(f"Found {len(heatmap_data)} heatmaps from {len(model_dirs)} models")
    
    # Build HTML
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Grad-CAM Heatmap Dashboard - 30 Models Comparison</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}
        
        .container {{
            max-width: 1600px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            padding: 30px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }}
        
        h1 {{
            text-align: center;
            color: #2d3748;
            margin-bottom: 10px;
            font-size: 2.5em;
        }}
        
        .subtitle {{
            text-align: center;
            color: #718096;
            margin-bottom: 30px;
            font-size: 1.1em;
        }}
        
        .stats {{
            display: flex;
            justify-content: center;
            gap: 40px;
            margin-bottom: 30px;
            flex-wrap: wrap;
        }}
        
        .stat-box {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px 30px;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        }}
        
        .stat-number {{
            font-size: 2em;
            font-weight: bold;
        }}
        
        .stat-label {{
            font-size: 0.9em;
            opacity: 0.9;
        }}
        
        .filters {{
            background: #f7fafc;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 30px;
            display: flex;
            gap: 15px;
            flex-wrap: wrap;
            align-items: center;
        }}
        
        .filter-group {{
            display: flex;
            flex-direction: column;
            gap: 5px;
        }}
        
        .filter-group label {{
            font-weight: 600;
            color: #4a5568;
            font-size: 0.9em;
        }}
        
        select, input {{
            padding: 8px 12px;
            border: 2px solid #e2e8f0;
            border-radius: 6px;
            font-size: 0.95em;
            transition: all 0.3s;
        }}
        
        select:focus, input:focus {{
            outline: none;
            border-color: #667eea;
        }}
        
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }}
        
        .card {{
            background: white;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            transition: transform 0.3s, box-shadow 0.3s;
        }}
        
        .card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 10px 20px rgba(0,0,0,0.2);
        }}
        
        .card-img {{
            width: 100%;
            height: 250px;
            object-fit: cover;
            cursor: pointer;
        }}
        
        .card-body {{
            padding: 15px;
        }}
        
        .card-title {{
            font-weight: bold;
            color: #2d3748;
            margin-bottom: 8px;
            font-size: 1.1em;
        }}
        
        .card-info {{
            font-size: 0.9em;
            color: #718096;
            margin-bottom: 5px;
        }}
        
        .badge {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 0.85em;
            font-weight: 600;
            margin-right: 5px;
            margin-top: 5px;
        }}
        
        .badge-benign {{
            background: #c6f6d5;
            color: #22543d;
        }}
        
        .badge-malignant {{
            background: #fed7d7;
            color: #742a2a;
        }}
        
        .badge-dataset {{
            background: #e6fffa;
            color: #234e52;
        }}
        
        .modal {{
            display: none;
            position: fixed;
            z-index: 1000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            background: rgba(0,0,0,0.9);
            align-items: center;
            justify-content: center;
        }}
        
        .modal-content {{
            max-width: 90%;
            max-height: 90%;
        }}
        
        .close {{
            position: absolute;
            top: 20px;
            right: 40px;
            color: white;
            font-size: 40px;
            font-weight: bold;
            cursor: pointer;
        }}
        
        .no-results {{
            text-align: center;
            padding: 60px;
            color: #718096;
            font-size: 1.2em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Grad-CAM Heatmap Dashboard</h1>
        <p class="subtitle">Visual Interpretation of Model Predictions - Breast Cancer Classification</p>
        
        <div class="stats">
            <div class="stat-box">
                <div class="stat-number" id="total-models">{len(model_dirs)}</div>
                <div class="stat-label">Models</div>
            </div>
            <div class="stat-box">
                <div class="stat-number" id="total-heatmaps">{len(heatmap_data)}</div>
                <div class="stat-label">Heatmaps</div>
            </div>
            <div class="stat-box">
                <div class="stat-number">6</div>
                <div class="stat-label">Architectures</div>
            </div>
        </div>
        
        <div class="filters">
            <div class="filter-group">
                <label>Architecture</label>
                <select id="filter-architecture">
                    <option value="">All Architectures</option>
                    <option value="ResNet18">ResNet18</option>
                    <option value="DenseNet">DenseNet121</option>
                    <option value="ViT">Vision Transformer</option>
                    <option value="Swin">Swin Transformer</option>
                    <option value="BioViT">BioViT</option>
                    <option value="MedViT">MedViT</option>
                </select>
            </div>
            
            <div class="filter-group">
                <label>Training Dataset</label>
                <select id="filter-dataset">
                    <option value="">All Datasets</option>
                    <option value="ORIGINAL">ORIGINAL</option>
                    <option value="DCGAN">DCGAN</option>
                    <option value="DCGAN_UPSCALED">DCGAN UPSCALED</option>
                    <option value="DDPM">DDPM</option>
                    <option value="DDPM_UPSCALED">DDPM UPSCALED</option>
                </select>
            </div>
            
            <div class="filter-group">
                <label>Label</label>
                <select id="filter-label">
                    <option value="">All Labels</option>
                    <option value="Benign">Benign</option>
                    <option value="Malignant">Malignant</option>
                </select>
            </div>
            
            <div class="filter-group">
                <label>Source Dataset</label>
                <select id="filter-source">
                    <option value="">All Sources</option>
                    <option value="ORIGINAL">ORIGINAL</option>
                    <option value="DCGAN">DCGAN</option>
                    <option value="DCGAN">DCGAN_UPSCALED</option>
                    <option value="DDPM">DDPM</option>
                    <option value="DDPM">DDPM_UPSCALED</option>
                </select>
            </div>
            
            <div class="filter-group">
                <label>Search</label>
                <input type="text" id="search" placeholder="Search model or filename...">
            </div>
        </div>
        
        <div class="grid" id="heatmap-grid"></div>
        <div class="no-results" id="no-results" style="display: none;">No heatmaps match your filters</div>
    </div>
    
    <div id="modal" class="modal">
        <span class="close">&times;</span>
        <img class="modal-content" id="modal-img">
    </div>
    
    <script>
        const heatmaps = {json.dumps(heatmap_data)};
        
        function renderHeatmaps(filteredData) {{
            const grid = document.getElementById('heatmap-grid');
            const noResults = document.getElementById('no-results');
            
            if (filteredData.length === 0) {{
                grid.style.display = 'none';
                noResults.style.display = 'block';
                return;
            }}
            
            grid.style.display = 'grid';
            noResults.style.display = 'none';
            
            grid.innerHTML = filteredData.map(h => `
                <div class="card">
                    <img src="${{h.path}}" alt="${{h.filename}}" class="card-img" onclick="openModal('${{h.path}}')">
                    <div class="card-body">
                        <div class="card-title">${{h.architecture}}</div>
                        <div class="card-info">Training: ${{h.dataset}}</div>
                        <div class="card-info">Source: ${{h.source}}</div>
                        <div>
                            <span class="badge badge-${{h.label.toLowerCase()}}">${{h.label}}</span>
                            <span class="badge badge-dataset">${{h.source}}</span>
                        </div>
                    </div>
                </div>
            `).join('');
        }}
        
        function filterHeatmaps() {{
            const arch = document.getElementById('filter-architecture').value.toLowerCase();
            const dataset = document.getElementById('filter-dataset').value.toLowerCase();
            const label = document.getElementById('filter-label').value.toLowerCase();
            const source = document.getElementById('filter-source').value.toLowerCase();
            const search = document.getElementById('search').value.toLowerCase();
            
            const filtered = heatmaps.filter(h => {{
                return (!arch || h.architecture.toLowerCase().includes(arch)) &&
                       (!dataset || h.dataset.toLowerCase().includes(dataset)) &&
                       (!label || h.label.toLowerCase().includes(label)) &&
                       (!source || h.source.toLowerCase().includes(source)) &&
                       (!search || h.model.toLowerCase().includes(search) || h.filename.toLowerCase().includes(search));
            }});
            
            renderHeatmaps(filtered);
        }}
        
        document.getElementById('filter-architecture').addEventListener('change', filterHeatmaps);
        document.getElementById('filter-dataset').addEventListener('change', filterHeatmaps);
        document.getElementById('filter-label').addEventListener('change', filterHeatmaps);
        document.getElementById('filter-source').addEventListener('change', filterHeatmaps);
        document.getElementById('search').addEventListener('input', filterHeatmaps);
        
        function openModal(imgSrc) {{
            const modal = document.getElementById('modal');
            const modalImg = document.getElementById('modal-img');
            modal.style.display = 'flex';
            modalImg.src = imgSrc;
        }}
        
        document.querySelector('.close').onclick = function() {{
            document.getElementById('modal').style.display = 'none';
        }}
        
        window.onclick = function(event) {{
            const modal = document.getElementById('modal');
            if (event.target == modal) {{
                modal.style.display = 'none';
            }}
        }}
        
        // Initial render
        renderHeatmaps(heatmaps);
    </script>
</body>
</html>
"""
    
    # Save HTML
    html_path = Path(output_dir) / "GRADCAM_DASHBOARD.html"
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"\nHTML Dashboard created: {html_path}")
    print(f"Total models: {len(model_dirs)}")
    print(f"Total heatmaps: {len(heatmap_data)}")
    return html_path

if __name__ == "__main__":
    output_dir = "./gradcam_outputs"
    build_html_dashboard(output_dir)
