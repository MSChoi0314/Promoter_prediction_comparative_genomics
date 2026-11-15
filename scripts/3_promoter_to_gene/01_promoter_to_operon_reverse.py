import csv
import os
import sys
import argparse
import re
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

PROMOTER_LENGTH = 100
MAX_DISTANCE_TO_CDS = 101
NUM_THREADS = 4


def setup_logging(output_path: str):
    base = os.path.splitext(output_path)[0]
    log_path = base + ".log"
    return log_path


def log_message(msg: str, log_file: Optional[str] = None):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full = f"[{timestamp}] {msg}"
    print(full)
    if log_file:
        try:
            with open(log_file, "a") as lf:
                lf.write(full + "\n")
        except Exception:
            print(f"[{timestamp}] ERROR: Failed writing to log file.", file=sys.stderr)


def parse_attributes(attributes_str: str) -> Dict[str, str]:
    attrs: Dict[str, str] = {}
    for item in attributes_str.split(";"):
        if "=" in item:
            key, value = item.split("=", 1)
            attrs[key.strip()] = value.strip()
    return attrs


def load_gff(database_dir: str, accession: str, log_file: Optional[str] = None) -> List[Dict]:
    gff_path = os.path.join(database_dir, accession, "genomic.gff")
    if not os.path.exists(gff_path):
        msg = f"ERROR: GFF file not found: {gff_path}"
        log_message(msg, log_file)
        raise FileNotFoundError(msg)

    features: List[Dict] = []
    with open(gff_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 9:
                continue
            try:
                feature = {
                    "seqid": parts[0],
                    "source": parts[1],
                    "type": parts[2],
                    "start": int(parts[3]),
                    "end": int(parts[4]),
                    "score": parts[5],
                    "strand": parts[6],
                    "phase": parts[7],
                    "attributes": parts[8],
                }
                features.append(feature)
            except (ValueError, IndexError):
                continue
    return features


def get_scaffold_length(gff_features: List[Dict], scaffold: str, raw_id: str) -> int:
    for feature in gff_features:
        if feature["seqid"] == scaffold and feature["type"] == "region":
            return feature["end"]

    raise RuntimeError(
        f"Missing 'region' row for scaffold '{scaffold}' in GFF. "
        f"Cannot calculate strand conversion. Failed on sequence: {raw_id}"
    )


def load_protein_faa(database_dir: str, accession: str) -> Dict[str, str]:
    faa_path = os.path.join(database_dir, accession, "protein.faa")
    if not os.path.exists(faa_path):
        raise FileNotFoundError(f"Protein file not found: {faa_path}")

    proteins: Dict[str, str] = {}
    current_id: Optional[str] = None
    current_seq: List[str] = []

    def clean_protein_id(raw_id: str) -> str:
        if raw_id.startswith("pgaptmp_"):
            return raw_id
        return re.sub(r'_(\d+)$', '', raw_id)

    with open(faa_path, 'r') as f:
        for line in f:
            line = line.rstrip()
            if line.startswith('>'):
                if current_id:
                    proteins[current_id] = ''.join(current_seq)

                header_content = line[1:]

                if 'pgaptmp' in header_content:
                    match = re.search(r'(pgaptmp_\S+)', header_content)
                    if match:
                        raw_id = match.group(1)
                        current_id = clean_protein_id(raw_id)
                    else:
                        current_id = None
                else:
                    header_parts = header_content.split()
                    if header_parts:
                        raw_id = header_parts[0]
                        current_id = clean_protein_id(raw_id)
                    else:
                        current_id = None

                current_seq = []
            else:
                if current_id:
                    current_seq.append(line)

    if current_id:
        proteins[current_id] = ''.join(current_seq)

    return proteins


def parse_operons_csv(operon_dir: str, accession: str, log_file: Optional[str] = None) -> Dict[str, List[str]]:
    operon_path = os.path.join(operon_dir, f"{accession}_operon", f"{accession}_list_of_operons.csv")
    if not os.path.exists(operon_path):
        msg = f"ERROR: Operon file not found: {operon_path}"
        log_message(msg, log_file)
        raise FileNotFoundError(msg)

    operons_list: List[List[str]] = []
    current_operon: List[str] = []
    gene_to_operon: Dict[str, List[str]] = {}

    with open(operon_path, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or all(not cell.strip() for cell in row):
                continue
            if len(row) == 1 and row[0].strip().isdigit():
                if current_operon:
                    operons_list.append(current_operon)
                current_operon = []
                continue
            if len(row) > 1:
                gene_or_id = row[1].strip()
                if gene_or_id:
                    current_operon.append(gene_or_id)

    if current_operon:
        operons_list.append(current_operon)

    for operon_genes in operons_list:
        for gene in operon_genes:
            gene_to_operon[gene] = operon_genes

    return gene_to_operon


def find_target_cds(
    gff_features: List[Dict],
    scaffold: str,
    target_pos: int,
) -> Optional[Dict]:
    candidates: List[Dict] = []
    for feature in gff_features:
        if (
            feature["seqid"] != scaffold
            or feature["type"] != "CDS"
            or feature["strand"] != "-"
            or feature["end"] > target_pos
        ):
            continue

        candidates.append(feature)

    if not candidates:
        return None

    cds_feature = max(candidates, key=lambda x: x["end"])
    return cds_feature


def get_gene_name_and_id(gff_features: List[Dict], cds_feature: Dict) -> Tuple[Optional[str], str]:
    attrs = parse_attributes(cds_feature["attributes"])
    if "ID" not in attrs:
        raise ValueError(f"CDS feature missing ID attribute: {cds_feature}")
    feature_id = attrs["ID"]
    gene_name = attrs.get("gene", None)
    if not gene_name:
        return None, feature_id

    count_above = 0
    for feature in gff_features:
        if feature["type"] != "CDS":
            continue
        if feature is cds_feature:
            break
        feature_attrs = parse_attributes(feature["attributes"])
        if feature_attrs.get("gene") == gene_name:
            count_above += 1

    if count_above > 0:
        gene_name_numbered = f"{gene_name}_{count_above + 1}"
    else:
        gene_name_numbered = gene_name
    return gene_name_numbered, feature_id


def parse_gene_name_with_number(gene_name_with_number: str) -> Tuple[str, Optional[int]]:
    if "_" not in gene_name_with_number:
        return gene_name_with_number, None
    base_name, suffix = gene_name_with_number.rsplit("_", 1)
    try:
        number = int(suffix)
        if number > 0:
            return base_name, number
        return gene_name_with_number, None
    except ValueError:
        return gene_name_with_number, None


def format_singleton_output(cds_id: str, gene_name: Optional[str]) -> str:
    if gene_name:
        return f"{gene_name}({cds_id})"
    else:
        return cds_id


def build_cds_gene_mappings(gff_features: List[Dict]) -> Tuple[Dict[str, Optional[str]], Dict[str, str], Dict[str, List[str]]]:
    id_to_gene_name: Dict[str, Optional[str]] = {}
    gene_name_to_id: Dict[str, str] = {}
    gene_name_to_all_ids: Dict[str, List[str]] = {}

    for feature in gff_features:
        if feature["type"] != "CDS":
            continue
        attrs = parse_attributes(feature["attributes"])
        feature_id = attrs.get("ID")
        gene_name_attr = attrs.get("gene")

        if not feature_id or not (feature_id.startswith("cds-WP_") or feature_id.startswith("cds-pgaptmp_")):
            continue

        id_to_gene_name[feature_id] = gene_name_attr

        if gene_name_attr:
            if gene_name_attr not in gene_name_to_id:
                gene_name_to_id[gene_name_attr] = feature_id
            if gene_name_attr not in gene_name_to_all_ids:
                gene_name_to_all_ids[gene_name_attr] = []
            gene_name_to_all_ids[gene_name_attr].append(feature_id)

    return id_to_gene_name, gene_name_to_id, gene_name_to_all_ids


def get_operon_output_string(
    gff_features: List[Dict],
    gene_to_operon: Dict[str, List[str]],
    search_key: str,
    gene_name: Optional[str],
    cds_id: str,
) -> Tuple[str, bool]:

    if search_key not in gene_to_operon:
        if not (cds_id.startswith("cds-WP_") or cds_id.startswith("cds-pgaptmp_")):
            return "", True
        if cds_id.startswith("cds-WP_") and is_wp_pseudogene_cds(gff_features, cds_id):
            return "", True
        if cds_id.startswith("cds-pgaptmp_") and is_pgaptmp_pseudogene_cds(gff_features, cds_id):
            return "", True

        singleton_output = format_singleton_output(cds_id, gene_name)
        return singleton_output, True

    operon_genes = gene_to_operon[search_key]
    id_to_gene_name, gene_name_to_id, gene_name_to_all_ids = build_cds_gene_mappings(gff_features)

    formatted_genes: List[str] = []
    seen_wp_ids: set = set()

    for gene in operon_genes:
        cds_full: Optional[str] = None
        wp_id: Optional[str] = None
        gene_name_local: Optional[str] = None

        if gene.startswith("cds-WP_") or gene.startswith("cds-pgaptmp_"):
            cds_full = gene
            wp_id = gene.replace("cds-", "")
            gene_name_local = id_to_gene_name.get(cds_full)

        elif gene.startswith("cds-"):
            continue

        else:
            base_gene_name, number = parse_gene_name_with_number(gene)
            all_cds_for_gene = gene_name_to_all_ids.get(base_gene_name)
            if not all_cds_for_gene:
                continue
            if number is None:
                cds_full = all_cds_for_gene[0]
                gene_name_local = base_gene_name
            else:
                if 1 <= number <= len(all_cds_for_gene):
                    index = number - 1
                    cds_full = all_cds_for_gene[index]
                    gene_name_local = base_gene_name
                else:
                    continue
            wp_id = cds_full.replace("cds-", "")

        if not cds_full or not wp_id:
            continue

        if cds_full.startswith("cds-pgaptmp_") and is_pgaptmp_pseudogene_cds(gff_features, cds_full):
            continue

        if wp_id in seen_wp_ids:
            continue
        seen_wp_ids.add(wp_id)

        if gene_name_local:
            formatted_output = f"{gene_name_local}({cds_full})"
        else:
            formatted_output = cds_full
        formatted_genes.append(formatted_output)

    if not formatted_genes:
        return "", True

    return ", ".join(formatted_genes), False


def extract_ids_from_operon_string(operon_string: str) -> List[str]:
    cds_ids: List[str] = []
    seen_ids: set = set()
    genes = [g.strip() for g in operon_string.split(",") if g.strip()]

    for gene in genes:
        cds_full: Optional[str] = None
        if "(" in gene and ")" in gene:
            start_idx = gene.rfind("(")
            end_idx = gene.rfind(")")
            if start_idx < end_idx:
                extracted = gene[start_idx + 1 : end_idx].strip()
                if extracted.startswith("cds-WP_") or extracted.startswith("cds-pgaptmp_"):
                    cds_full = extracted
        elif gene.startswith("cds-WP_") or gene.startswith("cds-pgaptmp_"):
            cds_full = gene.strip()

        if not cds_full:
            continue
        if cds_full in seen_ids:
            continue
        seen_ids.add(cds_full)
        cds_ids.append(cds_full)

    return cds_ids


def get_protein_sequences(
    proteins: Dict[str, str],
    operon_string: str,
    scaffold: str,
    promoter_start: int,
    gff_features: List[Dict],
) -> List[str]:
    cds_ids = extract_ids_from_operon_string(operon_string)
    result_sequences: List[str] = []

    for cds_full in cds_ids:
        wp_id = cds_full.replace("cds-", "")
        if "-" in wp_id:
            wp_id_base = wp_id.rsplit("-", 1)[0]
        else:
            wp_id_base = wp_id

        if not wp_id_base.startswith("pgaptmp_"):
            wp_id_base = re.sub(r'_(\d+)$', '', wp_id_base)

        if wp_id_base not in proteins:
            if wp_id_base.startswith("pgaptmp_"):
                is_pseudo = check_pgaptmp_pseudo(gff_features, wp_id_base)
                if is_pseudo:
                    continue
            else:
                raise ValueError(
                    f"Protein ID '{wp_id}' (base: '{wp_id_base}') not found in protein.faa\n"
                    f" Processing row: scaffold={scaffold}, start={promoter_start}"
                )
        else:
            result_sequences.append(proteins[wp_id_base])

    return result_sequences


def parse_input_id(raw: str) -> Tuple[str, str, int]:
    raw_stripped = raw[:-8]
    parts = raw_stripped.split("_")

    if len(parts) < 3:
        raise ValueError(f"Cannot parse id '{raw}' into accession, scaffold, and start")

    genomic_accession = "_".join(parts[0:2])
    promoter_start_str = parts[-1]
    scaffold_parts = parts[2:-1]
    scaffold_accession = "_".join(scaffold_parts)
    promoter_start = int(promoter_start_str)

    return genomic_accession, scaffold_accession, promoter_start


def is_wp_pseudogene_cds(gff_features: List[Dict], cds_id: str) -> bool:
    for feature in gff_features:
        if feature.get("type") != "CDS":
            continue
        attrs = parse_attributes(feature.get("attributes", ""))
        if attrs.get("ID") == cds_id:
            return attrs.get("pseudo", "").lower() == "true"
    return False


def check_pgaptmp_pseudo(gff_features: List[Dict], pgaptmp_id: str) -> bool:
    target_cds_id = f"cds-{pgaptmp_id}"
    for feature in gff_features:
        if feature["type"] != "CDS":
            continue
        attrs = parse_attributes(feature["attributes"])
        if attrs.get("ID") == target_cds_id or attrs.get("locus_tag") == pgaptmp_id:
            if attrs.get("pseudo", "").lower() == "true":
                return True
            else:
                raise ValueError(
                    f"pgaptmp protein '{pgaptmp_id}' not found in protein.faa "
                    f"but its CDS in genomic.gff is NOT marked pseudo=true."
                )
    raise ValueError(
        f"pgaptmp protein '{pgaptmp_id}' not found in protein.faa "
        f"and no matching CDS entry found in genomic.gff at all."
    )


def is_pgaptmp_pseudogene_cds(gff_features: List[Dict], cds_full: str) -> bool:
    if not cds_full.startswith("cds-pgaptmp_"):
        return False

    pgaptmp_id = cds_full.replace("cds-", "")
    for feature in gff_features:
        if feature.get("type") != "CDS":
            continue
        attrs = parse_attributes(feature.get("attributes", ""))
        if attrs.get("ID") == cds_full or attrs.get("locus_tag") == pgaptmp_id:
            return attrs.get("pseudo", "").lower() == "true"
    return False


def process_row(
    row_index: int,
    raw_id: str,
    header_name: str,
    database_dir: str,
    operon_dir: str,
    log_file: str,
) -> Optional[Tuple[List[str], int]]:
    if not raw_id.endswith("_reverse"):
        log_message(f"WARNING: Row {row_index} ('{raw_id}') missing '_reverse' tag, skipping...", log_file)
        return None

    try:
        accession, scaffold, promoter_start = parse_input_id(raw_id)
    except Exception as e:
        raise ValueError(f"Row {row_index}: error parsing '{raw_id}': {e}") from e

    gff_features = load_gff(database_dir, accession)
    proteins = load_protein_faa(database_dir, accession)
    gene_to_operon = parse_operons_csv(operon_dir, accession)

    scaffold_length = get_scaffold_length(gff_features, scaffold, raw_id)

    # promoter_start is counted on the reverse complement; convert to + strand coordinates
    converted_promoter_location = scaffold_length - promoter_start + 1
    target_pos = converted_promoter_location - PROMOTER_LENGTH

    cds_feature = find_target_cds(gff_features, scaffold, target_pos)

    if cds_feature is None:
        return None

    distance = target_pos - cds_feature["end"]

    if distance > MAX_DISTANCE_TO_CDS:
        attrs = parse_attributes(cds_feature["attributes"])
        cds_id = attrs.get("ID", "Unknown_ID")
        log_message(
            f"INFO: Discarding CDS {cds_id} for promoter '{raw_id}' - distance {distance}bp exceeds {MAX_DISTANCE_TO_CDS}bp threshold.",
            log_file
        )
        return None

    gene_name_numbered, feature_id = get_gene_name_and_id(gff_features, cds_feature)
    search_key = gene_name_numbered if gene_name_numbered else feature_id

    result = get_operon_output_string(
        gff_features, gene_to_operon, search_key, gene_name_numbered, feature_id
    )

    if result is None:
        return None

    operon_string, is_singleton = result

    if not operon_string.strip():
        return None

    protein_sequences = get_protein_sequences(
        proteins, operon_string, scaffold, promoter_start, gff_features,
    )

    out_row = [raw_id, str(distance), operon_string]
    out_row.extend(protein_sequences)
    return out_row, len(protein_sequences)


def main():
    parser = argparse.ArgumentParser(description="Map promoters to downstream operon proteins (- strand)")
    parser.add_argument("--input_csv", required=True, help="Input CSV file")
    parser.add_argument("--db_dir", required=True, help="Database directory with <accession>/ subdirs")
    parser.add_argument("--operondb", required=True, help="Operon directory with <accession>_operon/ subdirs")
    parser.add_argument("--output", required=True, help="Output CSV file")
    args = parser.parse_args()

    input_csv = args.input_csv
    database_dir = args.db_dir
    operon_dir = args.operondb
    output_csv = args.output

    if not os.path.isfile(input_csv):
        print(f"ERROR: input_csv is not a file: {input_csv}")
        sys.exit(1)
    if not os.path.isdir(database_dir):
        print(f"ERROR: db_dir is not a directory: {database_dir}")
        sys.exit(1)
    if not os.path.isdir(operon_dir):
        print(f"ERROR: operondb is not a directory: {operon_dir}")
        sys.exit(1)

    log_file = setup_logging(output_csv)
    log_message("Start", log_file)
    log_message(f"Input CSV: {input_csv}", log_file)
    log_message(f"DB dir: {database_dir}", log_file)
    log_message(f"Operon dir: {operon_dir}", log_file)
    log_message(f"Output CSV: {output_csv}", log_file)
    log_message(f"PROMOTER_LENGTH: {PROMOTER_LENGTH}", log_file)
    log_message(f"MAX_DISTANCE_TO_CDS: {MAX_DISTANCE_TO_CDS}", log_file)
    log_message(f"NUM_THREADS: {NUM_THREADS}", log_file)

    rows: List[Tuple[int, str]] = []
    original_header_first_col = None

    with open(input_csv, "r") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            log_message("ERROR: Input CSV is empty.", log_file)
            sys.exit(1)

        if not header:
            log_message("ERROR: Input CSV header is empty.", log_file)
            sys.exit(1)
        original_header_first_col = header[0]

        for idx, row in enumerate(reader, start=1):
            if not row or all(not c.strip() for c in row):
                continue
            raw_id = row[0].strip()
            if not raw_id:
                log_message(f"Row {idx}: Empty first column, skipping.", log_file)
                continue
            rows.append((idx, raw_id))

    log_message(f"Total non-empty data rows to process: {len(rows)}", log_file)

    results: List[Optional[Tuple[List[str], int]]] = [None] * len(rows)

    with ProcessPoolExecutor(max_workers=NUM_THREADS) as executor:
        future_to_index = {}
        for i, (row_index, raw_id) in enumerate(rows):
            future = executor.submit(
                process_row,
                row_index,
                raw_id,
                original_header_first_col,
                database_dir,
                operon_dir,
                log_file,
            )
            future_to_index[future] = i

        for future in as_completed(future_to_index):
            i = future_to_index[future]
            row_index, raw_id = rows[i]
            try:
                res = future.result()
                results[i] = res
            except Exception as e:
                log_message(f"Fatal error in worker for row {row_index} ('{raw_id}'): {e}", log_file)
                log_message("Aborting due to fatal error.", log_file)
                sys.exit(1)

    output_rows: List[List[str]] = []
    max_protein_count = 0
    for res in results:
        if res is None:
            continue
        row, n_proteins = res
        max_protein_count = max(max_protein_count, n_proteins)
        output_rows.append(row)

    log_message(f"Total rows kept: {len(output_rows)}", log_file)

    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        header_row = [
            original_header_first_col,
            "distance_from_promoter_to_cds",
            "operon_protein_names",
        ]
        for i in range(1, max_protein_count + 1):
            header_row.append(f"protein_seq_{i}")
        writer.writerow(header_row)

        for row in output_rows:
            while len(row) < len(header_row):
                row.append("")
            writer.writerow(row)

    log_message("Done", log_file)


if __name__ == "__main__":
    main()
