import json

def create_cypher_queries(json_file_path):
    """
    Reads the JSON file and generates Cypher queries to:
      - Create a Chapter node for each chapter.
      - Create a Concept node for each unique concept across all chapters.
      - Create relationships (TALKS_ABOUT) from each Chapter node to its associated Concept nodes.
    """
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    queries = []

    # Dictionary to track unique concepts and assign each variable a name
    concept_vars = {}
    concept_counter = 1

    # First pass: collecting all unique concepts from all chapters
    for chapter in data:
        for concept in chapter.get('concepts', []):
            concept = concept.replace("'", "")
            if concept not in concept_vars:
                # Create a variable name by using a prefix and a counter (also removing spaces)
                var_name = f"cpt{concept_counter}"
                concept_vars[concept] = var_name
                concept_counter += 1

    # Creating chapter nodes
    for idx, chapter in enumerate(data, start=1):
        chap_title = chapter.get("title", "untitled")
        chap_number = chapter.get("chapter", idx)
        # Create variable name for chapter: c1, c2, etc.
        chapter_var = f"c{idx}"
        chapter_query = f"CREATE ({chapter_var}:Chapter {{title: '{chap_title}', chapterNumber: '{chap_number}'}})"
        queries.append(chapter_query)

    # Creating concept nodes
    for concept, var_name in concept_vars.items():
        concept = concept.replace("'", "")
        concept_query = f"CREATE ({var_name}:Concept {{name: '{concept}'}})"
        queries.append(concept_query)

    # Creating relationships: For each chapter, for each concept in its "concepts" list, create a TALKS_ABOUT relationship.
    for idx, chapter in enumerate(data, start=1):
        chapter_var = f"c{idx}"
        for concept in chapter.get("concepts", []):
            concept = concept.replace("'", "")
            concept_var = concept_vars[concept]
            relationship_query = f"CREATE ({chapter_var})-[:CONTAINS]->({concept_var})"
            queries.append(relationship_query)

    # Join all queries into a single string separated by newlines
    full_query = "\n".join(queries)
    return full_query

    

