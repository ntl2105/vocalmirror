import os

project_structure = {
    "applied_tool/data/": [],
    "applied_tool/features/": [
        "extract_pitch_f0.py",
        "extract_formants_f1f2.py",
        "extract_duration.py",
        "extract_voice_quality.py"
    ],
    "applied_tool/learner_feedback/": [
        "feedback_rules.py",
        "generate_feedback.py"
    ],
    "applied_tool/prototype/": [
        "cli_tool.py",
        "notebook_demo.ipynb"
    ],
    "eda_research/": [
        "vowel_space_analysis.ipynb",
        "pitch_contour_clustering.ipynb",
        "umap_acoustic_space.ipynb",
        "dialect_distance_heatmap.ipynb",
        "learner_projection.ipynb"
    ]
}

for folder, files in project_structure.items():
    os.makedirs(folder, exist_ok=True)
    for f in files:
        with open(os.path.join(folder, f), 'w') as file:
            file.write(f"# {f.replace('_', ' ').capitalize().replace('.py', '').replace('.ipynb', '')}\n")
            