import argparse
import csv
import os
import subprocess

from Bio import SeqIO


def short_id(header):
    return header.split()[0] if header else ""


def run(cmd):
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True).stdout


def self_bit_scores(blastp, query_file, threads):
    out = run([blastp, "-query", query_file, "-subject", query_file,
               "-outfmt", "10 qseqid sseqid bitscore", "-num_threads", threads])
    best = {}
    for row in csv.reader(out.splitlines()):
        if not row:
            continue
        q, s, bits = short_id(row[0]), short_id(row[1]), float(row[2])
        if q == s and bits > best.get(q, 0.0):
            best[q] = bits
    return best


def run_blast(query_file, root_dir, output_csv, blastp, makeblastdb, threads):
    if not os.path.exists(query_file):
        print(f"{query_file} not found")
        return
    if not os.path.exists(root_dir):
        print(f"{root_dir} not found")
        return

    lengths = {}
    order = []
    for rec in SeqIO.parse(query_file, "fasta"):
        qid = short_id(rec.id)
        lengths[qid] = len(rec.seq)
        order.append(qid)

    self_bits = self_bit_scores(blastp, query_file, threads)
    hits = {q: {} for q in order}
    accessions = []

    for acc in sorted(os.listdir(root_dir)):
        acc_dir = os.path.join(root_dir, acc)
        faa = os.path.join(acc_dir, "protein.faa")
        if not (os.path.isdir(acc_dir) and os.path.exists(faa)):
            continue
        print(acc)
        accessions.append(acc)

        subprocess.run([makeblastdb, "-in", faa, "-dbtype", "prot"],
                       check=True, stdout=subprocess.DEVNULL)
        out = run([blastp, "-query", query_file, "-db", faa,
                   "-outfmt", "10 qseqid sseqid evalue bitscore", "-num_threads", threads])

        for row in csv.reader(out.splitlines()):
            if not row:
                continue
            q, evalue, bits = short_id(row[0]), row[2], float(row[3])
            if q in hits:
                cur = hits[q].get(acc)
                if cur is None or bits > cur["bitscore"]:
                    hits[q][acc] = {"evalue": evalue, "bitscore": bits}

        for ext in (".phr", ".pin", ".psq"):
            if os.path.exists(faa + ext):
                os.remove(faa + ext)

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Query Header", "Length", "Self-Bit-Score"] + accessions)
        for q in order:
            base = [lengths[q], self_bits.get(q, 0.0)]
            e_row = [f"{q}_e-value"] + base
            b_row = [f"{q}_bit-score"] + base
            for acc in accessions:
                h = hits[q].get(acc)
                e_row.append(h["evalue"] if h else "NA")
                b_row.append(h["bitscore"] if h else "NA")
            writer.writerow(e_row)
            writer.writerow(b_row)
    print(f"Wrote {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Best BLASTP hit of each query protein in every genome")
    parser.add_argument("-q", "--query", required=True, help="Query protein FASTA")
    parser.add_argument("-d", "--directory", required=True, help="Folder with <accession>/protein.faa")
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--blastp", default="../ncbi-blast-2.17.0+/bin/blastp")
    parser.add_argument("--makeblastdb", default="../ncbi-blast-2.17.0+/bin/makeblastdb")
    parser.add_argument("--threads", default="6")
    args = parser.parse_args()
    run_blast(args.query, args.directory, args.output, args.blastp, args.makeblastdb, args.threads)
