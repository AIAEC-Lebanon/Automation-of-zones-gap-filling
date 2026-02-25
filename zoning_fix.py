import psycopg2
import json

DB_CONFIG = {

}

z_query = """
SELECT
    z.id, z.code, z.description, z.node_ids, z.ordinance_document_id, z.document_node_ids
FROM zones z
WHERE z.id = %s;
"""

zg_query = """
SELECT
    zgz.zone_group_id
FROM zone_group_zones zgz
WHERE zgz.zone_id = %s;
"""

def copy_zone(location_id, base_zones, new_zone_name, new_zone_desc, relation):
    with psycopg2.connect(**DB_CONFIG) as conn:
        with conn.cursor() as cur:
            total_node_ids = []
            merged_document_node_ids = {}
            ordinance_document_id = None
            
            for base_zone_id in base_zones:
                cur.execute(z_query, (base_zone_id,))
                row = cur.fetchone()
                id, base_code, base_desc, node_ids, ord_id, doc_node_ids = row
                total_node_ids.extend(node_ids)
                
                # Use the first zone's ordinance_document_id
                if ordinance_document_id is None:
                    ordinance_document_id = ord_id
                
                # Merge document_node_ids JSON objects
                if doc_node_ids:
                    for doc_id, nodes in doc_node_ids.items():
                        if doc_id in merged_document_node_ids:
                            # Combine node lists and remove duplicates
                            merged_document_node_ids[doc_id] = list(set(merged_document_node_ids[doc_id] + nodes))
                        else:
                            merged_document_node_ids[doc_id] = nodes
            
            # Convert to JSON format: {ord_id: total_node_ids}
            document_node_ids = {str(ordinance_document_id): total_node_ids} if ordinance_document_id else None
            
            sql = """
                INSERT INTO zones (location_id, code, description, node_ids, ordinance_document_id, document_node_ids)
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING id;
            """
            if relation and len(base_zones) == 1:
                data = (
                    location_id,
                    new_zone_name,
                    new_zone_desc + f" [{relation} {base_code} {base_desc}]", 
                    total_node_ids,
                    ordinance_document_id,
                    json.dumps(document_node_ids) if document_node_ids else None
                )
            else:
                data = (
                    location_id,
                    new_zone_name,
                    new_zone_desc, 
                    total_node_ids,
                    ordinance_document_id,
                    json.dumps(document_node_ids) if document_node_ids else None
                )

            cur.execute(sql, data)
            new_id = cur.fetchone()[0]
            
            for base_zone_id in base_zones:
                cur.execute(zg_query, (base_zone_id,))
                rows = cur.fetchall()
                
                for (zg_id,) in rows:
                    sql = """
                        INSERT INTO zone_group_zones (zone_group_id, zone_id)
                        VALUES (%s, %s)
                        ON CONFLICT DO NOTHING;
                    """
                    
                    data = (
                        zg_id,
                        new_id
                    )

                    cur.execute(sql, data)
            conn.commit()

def insert_zone(location_id, new_zone_name, new_zone_desc):
    with psycopg2.connect(**DB_CONFIG) as conn:
        with conn.cursor() as cur:
            sql = """
                INSERT INTO zones (location_id, code, description, node_ids)
                VALUES (%s, %s, %s, %s)
                RETURNING id;
            """

            data = (
                location_id,
                new_zone_name,
                new_zone_desc, 
                []
            )
            cur.execute(sql, data)
            conn.commit()
        

# Replace the None with text if it is obsolete/legacy zoning, like "Replaced by"
# Otherwise copy zone can handle off names, extended, and combinations
# Location id, base zone ids, new code name, new code description, relation to original (None to directly copy)
copy_zone(4, [3666], "IND.PK.", "Industrial Park", None)
