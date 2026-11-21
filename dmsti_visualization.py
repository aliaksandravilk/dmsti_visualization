import re
import time
import json
import itertools
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from urllib.parse import quote_plus
from sklearn.decomposition import PCA
from sklearn.manifold import MDS

# I keep the group links here because these are the main staff sources for the task
GROUP_URLS = {
    "Artificial Intelligence Laboratory": "https://www.mii.lt/en/structure/scientific-subdivisions/artificial-intelligence-laboratory",
    "Blockchain and Quantum Technologies Group": "https://www.mii.lt/en/structure/scientific-subdivisions/blockchain-and-quantum-technologies-group",
    "Cognitive Computing Group": "https://www.mii.lt/en/structure/scientific-subdivisions/cognitive-computing-group",
    "Cyber-Social Systems Engineering Group": "https://www.mii.lt/en/structure/scientific-subdivisions/cyber-social-systems-engineering-group",
    "Education Systems Group": "https://www.mii.lt/en/structure/scientific-subdivisions/education-systems-group",
    "Global Optimization Group": "https://www.mii.lt/en/structure/scientific-subdivisions/global-optimization-group",
    "Image and Signal Analysis Group": "https://www.mii.lt/en/structure/scientific-subdivisions/image-and-signal-analysis-group",
    "Intelligent Technologies Research Group": "https://www.mii.lt/en/structure/scientific-subdivisions/intelligent-technologies-research-group",
    "Interdisciplinary Statistical Research Group": "https://www.mii.lt/en/structure/scientific-subdivisions/interdisciplinary-statistical-research-group",
}

# colors are fixed so the group legend stays the same in PCA and MDS plots
GROUP_COLORS = {
    "Blockchain and Quantum Technologies Group": "tab:blue",
    "Artificial Intelligence Laboratory": "tab:orange",
    "Education Systems Group": "lightgreen",
    "Global Optimization Group": "lightcoral",
    "Intelligent Technologies Research Group": "saddlebrown",
    "Cyber-Social Systems Engineering Group": "tab:pink",
    "Cognitive Computing Group": "lightgray",
    "Interdisciplinary Statistical Research Group": "khaki",
    "Image and Signal Analysis Group": "skyblue",
}

HEADERS = {"User-Agent": "Mozilla/5.0"}
CACHE_FILE = "dmsti_publications_cache.json"

# sometimes names have titles, so I remove them before comparing authors
def clean_name(name):
    name = re.sub(r"\b(prof|habil|dr|assoc|professor|researcher|principal|senior|junior|doc)\b\.?", "", name, flags=re.I)
    name = re.sub(r"[^A-Za-zÀ-ž\s\-]", " ", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name

def normalize_name(name):
    name = clean_name(name).lower()
    replacements = {"ė": "e", "ę": "e", "ū": "u", "ų": "u", "š": "s", "č": "c", "ž": "z", "ą": "a", "į": "i"}
    for old, new in replacements.items():
        name = name.replace(old, new)
    return name

# eLABa often writes authors as "Surname, Name", so I change it to one format
def surname_name_to_name_surname(text):
    parts = [p.strip() for p in text.split(",")]
    if len(parts) >= 2:
        return clean_name(parts[1] + " " + parts[0])
    return clean_name(text)

# this function only downloads the page and turns HTML into plain text
def get_page_text(url):
    response = requests.get(url, headers=HEADERS, timeout=30)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    return soup.get_text("\n")

# on the group pages the staff name is usually placed before the department line
def extract_staff_from_group(group_name, url):
    text = get_page_text(url)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    staff = []
    for i, line in enumerate(lines):
        if line.startswith("Department:") and i > 0:
            possible_name = clean_name(lines[i - 1])
            if len(possible_name.split()) >= 2:
                staff.append({"name": possible_name, "group": group_name})
    unique = {}
    for person in staff:
        key = normalize_name(person["name"])
        unique[key] = person
    return list(unique.values())

def get_all_staff():
    all_staff = []
    for group_name, url in GROUP_URLS.items():
        print("Reading group:", group_name)
        try:
            group_staff = extract_staff_from_group(group_name, url)
            all_staff.extend(group_staff)
            # small pause so the website does not get too many requests at once
            time.sleep(1)
        except Exception as e:
            print("Could not read group:", group_name, e)
    staff_df = pd.DataFrame(all_staff)
    # some names can appear multiple times across groups
    staff_df = staff_df.drop_duplicates(subset=["name"])
    # sorting just makes the exported files easier to read later
    staff_df = staff_df.sort_values(["group", "name"]).reset_index(drop=True)
    return staff_df

# cache is used so I do not download the same pages every time
def load_cache():
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError:
        return {}

def save_cache(cache):
    with open(CACHE_FILE, "w", encoding="utf-8") as file:
        json.dump(cache, file, ensure_ascii=False, indent=2)

def parse_publications_for_author(author_name, cache):
    if author_name in cache:
        return cache[author_name]
    url = "https://elaba.mb.vu.lt/dmsti/?aut=" + quote_plus(author_name)
    print("Reading publications:", author_name)
    text = get_page_text(url)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    publications = []
    for line in lines:
        if "//" not in line:
            continue
        clean_line = re.sub(r"\s+", " ", line.lower()).strip()
        if len(clean_line) < 30:
            continue
        publications.append({
            "author_query": author_name,
            "publication_key": clean_line,
            "url": url
        })
    cache[author_name] = publications
    save_cache(cache)
    time.sleep(1)
    return publications

# this function goes through all researchers and collects their publication records
def collect_publications(staff_df):
    cache = load_cache()
    all_publications = []
    for name in staff_df["name"]:
        try:
            publications = parse_publications_for_author(name, cache)
            all_publications.extend(publications)
        except Exception as e:
            print("Could not read publications for:", name, e)
    return all_publications

# here I count co-authorship by checking which staff members have the same publication record
def build_coauthor_matrix(staff_df, publications):
    names = list(staff_df["name"])
    name_to_index = {}
    for i, name in enumerate(names):
        name_to_index[normalize_name(name)] = i
    publication_map = {}
    for publication in publications:
        publication_key = publication["publication_key"]
        author_name = normalize_name(publication["author_query"])
        if publication_key not in publication_map:
            publication_map[publication_key] = set()
        if author_name in name_to_index:
            publication_map[publication_key].add(name_to_index[author_name])
    matrix = np.zeros((len(names), len(names)), dtype=int)
    for staff_set in publication_map.values():
        if len(staff_set) < 2:
            continue
        for i, j in itertools.combinations(staff_set, 2):
            matrix[i][j] += 1
            matrix[j][i] += 1
    np.fill_diagonal(matrix, 0)
    print("Pairs with publications:", int(np.sum(matrix > 0) / 2))
    print("Max joint publications:", int(np.max(matrix)))
    return matrix

# after additional feedback, I also tried a second version of the visualization
# here I use the transformed 1/(a+1) matrix directly as dissimilarity
# without calculating extra Euclidean distances on top of it
# this keeps the original co-authorship distances more directly
def make_dissimilarity_matrix(coauthor_matrix):
    transformed = 1 / (coauthor_matrix.astype(float) + 1)
    np.fill_diagonal(transformed, 0)
    dissimilarity = transformed.copy()
    return transformed, dissimilarity

def get_short_label(name):
    parts = name.split()
    if len(parts) >= 2:
        return (parts[0][0] + parts[1][0]).upper()
    return name[:2].upper()

# this draws the final map: one point is one researcher
def plot_2d(coords, staff_df, title, filename):
    groups = sorted(staff_df["group"].unique())
    point_colors = [GROUP_COLORS.get(group, "black") for group in staff_df["group"]]
    plt.figure(figsize=(15, 10))
    plt.scatter(coords[:, 0], coords[:, 1], c=point_colors, s=80, alpha=0.9)
    for i, name in enumerate(staff_df["name"]):
        plt.text(coords[i, 0] + 0.005, coords[i, 1] + 0.005, get_short_label(name), fontsize=7)
    handles = []
    for group in groups:
        color = GROUP_COLORS.get(group, "black")
        handles.append(plt.Line2D([], [], marker="o", linestyle="", markersize=12, markerfacecolor=color, markeredgecolor=color, label=group))
    plt.legend(handles=handles, title="Groups", bbox_to_anchor=(1.03, 1), loc="upper left", fontsize=10, title_fontsize=12, frameon=True, borderpad=1)
    plt.title(title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

# this short report is saved separately so the result is easier to explain
def save_summary_report(staff_df, coauthor_matrix, smacof_stress, filename):
    total_people = len(staff_df)
    total_pairs = int(np.sum(coauthor_matrix > 0) / 2)
    strongest_value = int(np.max(coauthor_matrix))
    with open(filename, "w", encoding="utf-8") as file:
        file.write("DMSTI co-authorship visualization summary\n\n")
        file.write(f"Number of researchers: {total_people}\n")
        file.write(f"Number of research groups: {staff_df['group'].nunique()}\n")
        file.write(f"Number of researcher pairs with joint publications: {total_pairs}\n")
        file.write(f"Maximum number of joint publications for one pair: {strongest_value}\n")
        file.write(f"SMACOF stress value: {smacof_stress}\n\n")
        file.write("Method:\n")
        file.write("Publication data was retrieved from eLABa links.\n")
        file.write("Joint publications were filtered into an n x n integer matrix.\n")
        file.write("The matrix was transformed with 1 / (a + 1).\n")
        file.write("After additional feedback, the transformed matrix was used directly as the dissimilarity matrix.\n")
        file.write("No additional Euclidean distance calculation was applied on top of it.\n")
        file.write("The final 2D visualizations were created with PCA and SMACOF/MDS.\n")

def main():
    staff_df = get_all_staff()
    staff_df.to_csv("dmsti_staff.csv", index=False, encoding="utf-8-sig")
    publications = collect_publications(staff_df)
    # this direct dissimilarity matrix is used later for PCA and MDS
    coauthor_matrix = build_coauthor_matrix(staff_df, publications)
    coauthor_df = pd.DataFrame(coauthor_matrix, index=staff_df["name"], columns=staff_df["name"])
    coauthor_df.to_csv("coauthor_matrix.csv", encoding="utf-8-sig")
    transformed_matrix, dissimilarity = make_dissimilarity_matrix(coauthor_matrix)
    transformed_df = pd.DataFrame(transformed_matrix, index=staff_df["name"], columns=staff_df["name"])
    # I save the transformed matrix because it shows the 1/(a+1) step
    transformed_df.to_csv("transformed_matrix.csv", encoding="utf-8-sig")
    # this distance matrix is used later for PCA and MDS
    dissimilarity_df = pd.DataFrame(dissimilarity, index=staff_df["name"], columns=staff_df["name"])
    dissimilarity_df.to_csv("dissimilarity_matrix.csv", encoding="utf-8-sig")
    # PCA is required in the task, so I use it as the first comparison method
    pca = PCA(n_components=2, random_state=42)
    pca_coords = pca.fit_transform(dissimilarity)
    plot_2d(pca_coords, staff_df, "DMSTI staff visualized by joint publications, PCA", "dmsti_pca.png")
    # I use MDS here because this task is based on distances, not only on variance
    smacof = MDS(n_components=2, dissimilarity="precomputed", random_state=42, normalized_stress="auto")
    smacof_coords = smacof.fit_transform(dissimilarity)
    plot_2d(smacof_coords, staff_df, "DMSTI staff visualized by joint publications, SMACOF / MDS", "dmsti_smacof.png")
    print("SMACOF stress value:", smacof.stress_)
    save_summary_report(staff_df, coauthor_matrix, smacof.stress_, "summary_report.txt")
    print("\nDone.")
    print("Files created:")
    print("dmsti_staff.csv")
    print("coauthor_matrix.csv")
    print("transformed_matrix.csv")
    print("dissimilarity_matrix.csv")
    print("summary_report.txt")
    print("dmsti_pca.png")
    print("dmsti_smacof.png")

if __name__ == "__main__":
    main()