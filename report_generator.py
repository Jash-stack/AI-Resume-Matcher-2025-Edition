import pandas as pd
import tempfile
from jinja2 import Template

# --- Generate HTML Report ---
def generate_report(resume_text, skills, ranked_df, clustered_df):
    top_jobs = ranked_df[['Job Title', 'Company', 'Location', 'Match Score']].head(10).to_html(index=False)

    clusters = []
    for cluster_id in sorted(clustered_df['Cluster'].unique()):
        group = clustered_df[clustered_df['Cluster'] == cluster_id]
        html_table = group[['Job Title', 'Company', 'Location', 'Match Score']].head(5).to_html(index=False)
        clusters.append({"label": f"Cluster {cluster_id + 1}", "table": html_table})

    # HTML Template
    html_template = Template("""
    <html>
    <head>
        <style>
            body { font-family: 'Arial', sans-serif; padding: 20px; }
            h1, h2, h3 { color: #2c3e50; }
            table { width: 100%; border-collapse: collapse; margin-bottom: 20px; }
            th, td { border: 1px solid #ccc; padding: 8px; }
            th { background-color: #f2f2f2; }
        </style>
    </head>
    <body>
        <h1>Career Report</h1>
        <h2>Extracted Skills</h2>
        <p>{{ skills }}</p>
        <h2>Top Job Matches</h2>
        {{ top_jobs|safe }}
        <h2>Clustered Recommendations</h2>
        {% for c in clusters %}
            <h3>{{ c.label }}</h3>
            {{ c.table|safe }}
        {% endfor %}
    </body>
    </html>
    """)

    html_out = html_template.render(skills=", ".join(skills), top_jobs=top_jobs, clusters=clusters)
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
    with open(tmp_file.name, "w", encoding="utf-8") as f:
        f.write(html_out)
    return tmp_file.name
