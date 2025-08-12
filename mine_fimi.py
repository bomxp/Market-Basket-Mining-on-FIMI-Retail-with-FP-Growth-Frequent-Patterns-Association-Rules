# mine_fimi.py
# FP-Growth thuần Python cho dữ liệu FIMI (mỗi dòng 1 giao dịch, items cách nhau bởi khoảng trắng)
# Chức năng: frequent itemsets, lọc maximal itemsets, sinh association rules (confidence, lift, leverage, conviction)

import os
import math
import argparse
from collections import defaultdict, Counter
from itertools import combinations, chain

# =========================
# 1) Đọc giao dịch (FIMI)
# =========================
def load_transactions(path, limit=None):
    """
    Đọc file FIMI: mỗi dòng 1 giao dịch, item là số (hoặc chuỗi), cách nhau khoảng trắng.
    limit: chỉ đọc N dòng đầu (debug).
    """
    transactions = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit: break
            line = line.strip()
            if not line:
                continue
            items = line.split()
            # chuyển sang tuple hoặc list; ở đây để string luôn cũng được
            transactions.append(items)
    return transactions

# =========================
# 2) Cấu trúc FP-Tree
# =========================
class FPNode:
    __slots__ = ("item", "count", "parent", "children", "node_link")
    def __init__(self, item, parent):
        self.item = item
        self.count = 0
        self.parent = parent
        self.children = {}       # item -> FPNode
        self.node_link = None    # link tới node kế tiếp có cùng item (header table)

class FPTree:
    def __init__(self):
        self.root = FPNode(None, None)
        self.header_table = {}   # item -> [support_count, first_node]

    def add_transaction(self, items, count=1):
        """
        Thêm 1 giao dịch (đã được sắp theo tần suất giảm dần) vào cây.
        """
        current = self.root
        current.count += count  # đếm tổng giao dịch đi qua root (không bắt buộc nhưng tiện)
        for it in items:
            if it not in current.children:
                child = FPNode(it, current)
                current.children[it] = child
                # cập nhật header_table
                if it not in self.header_table:
                    self.header_table[it] = [0, None]
                # móc nối node-link
                head = self.header_table[it][1]
                if head is None:
                    self.header_table[it][1] = child
                else:
                    # đi đến node cuối rồi nối
                    while head.node_link is not None:
                        head = head.node_link
                    head.node_link = child
                current = child
            else:
                current = current.children[it]
            current.count += count

    def items_in_ascending_support(self):
        """
        Trả về danh sách item theo thứ tự tăng dần theo support trong cây hiện tại (dùng khi khai thác).
        """
        return sorted(self.header_table.keys(), key=lambda it: self.header_table[it][0])

# =========================
# 3) Xây FP-Tree ban đầu
# =========================
def build_fptree(transactions, min_support_count):
    """
    - Đếm tần suất item toàn cục
    - Lọc item < min_support_count
    - Sắp item trong mỗi giao dịch theo tần suất giảm dần (global)
    - Xây FP-Tree
    """
    # Đếm tần suất
    freq = Counter()
    for t in transactions:
        freq.update(t)

    # Lọc theo min_support_count
    freq = {it: c for it, c in freq.items() if c >= min_support_count}
    if not freq:
        return None, {}, 0

    # Thứ tự sắp xếp items theo tần suất giảm dần, tie-break theo item id để ổn định
    order = sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))
    rank = {it: i for i, (it, _) in enumerate(order)}

    # Xây cây
    tree = FPTree()
    # Trước khi add, header_table cần tổng support_count:
    for it, c in freq.items():
        tree.header_table[it] = [c, None]

    n_tx = 0
    for t in transactions:
        # giữ lại items đủ ngưỡng & sắp xếp theo rank
        filtered = [it for it in t if it in freq]
        if not filtered:
            continue
        filtered.sort(key=lambda x: rank[x])
        tree.add_transaction(filtered, count=1)
        n_tx += 1

    return tree, freq, n_tx

# =========================
# 4) Khai thác FP-Tree (đệ quy)
# =========================
def ascend_prefix_path(node):
    """
    Đi ngược từ node lên root để lấy 1 prefix path (không gồm node hiện tại).
    """
    path = []
    current = node.parent
    while current is not None and current.item is not None:
        path.append(current.item)
        current = current.parent
    path.reverse()
    return path

def conditional_pattern_base(tree, base_item):
    """
    Thu thập conditional pattern base cho 1 base_item:
    Trả về list các (prefix_path, count).
    """
    patterns = []
    # duyệt node-link của base_item
    node = tree.header_table[base_item][1]
    while node is not None:
        if node.count > 0:
            path = ascend_prefix_path(node)
            if path:
                patterns.append((path, node.count))
        node = node.node_link
    return patterns

def build_conditional_fptree(tree, base_item, min_support_count):
    """
    Xây conditional FP-Tree từ conditional pattern base của base_item.
    """
    cpb = conditional_pattern_base(tree, base_item)
    if not cpb:
        return None

    # Đếm tần suất trong CPB
    freq = Counter()
    for path, cnt in cpb:
        freq.update({it: cnt for it in path})

    # Lọc theo min_support_count
    freq = {it: c for it, c in freq.items() if c >= min_support_count}
    if not freq:
        return None

    order = sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))
    rank = {it: i for i, (it, _) in enumerate(order)}

    cond_tree = FPTree()
    # header_table tổng count cho từng item trong CPB (sau lọc)
    for it, c in freq.items():
        cond_tree.header_table[it] = [c, None]

    for path, cnt in cpb:
        filtered = [it for it in path if it in freq]
        if not filtered:
            continue
        filtered.sort(key=lambda x: rank[x])
        cond_tree.add_transaction(filtered, count=cnt)

    return cond_tree

def mine_tree(tree, prefix, min_support_count, freq_itemsets):
    """
    Khai thác cây hiện tại:
    - Duyệt item theo support tăng dần trong cây
    - Thêm (prefix ∪ {item}) vào freq_itemsets với support = header_table[item][0] (support trong cây)
    - Xây conditional tree và đệ quy
    """
    for item in tree.items_in_ascending_support():
        support = tree.header_table[item][0]
        new_itemset = tuple(sorted(prefix + [item]))
        freq_itemsets[frozenset(new_itemset)] = support

        # Conditional tree
        cond_tree = build_conditional_fptree(tree, item, min_support_count)
        if cond_tree is not None and cond_tree.header_table:
            mine_tree(cond_tree, list(new_itemset), min_support_count, freq_itemsets)

def fpgrowth(transactions, min_support):
    """
    Trả về:
      - freq_itemsets: dict{frozenset(items): support_count}
      - n_tx: số giao dịch
    """
    # min_support là tỉ lệ; convert sang count
    # Nếu min_support >= 1 coi là count luôn (đề phòng ai truyền nhầm)
    # nhưng mặc định ta xử lý tỉ lệ.
    # => Tính count sau khi build tree (n_tx nhận được từ build_fptree)
    # Cách: build 1 lần để lấy n_tx, rồi suy ra min_support_count, build lại (để chắc chắn).
    tmp_tree, _, n_tx = build_fptree(transactions, min_support_count=1)
    if tmp_tree is None:
        return {}, 0

    if min_support < 1:
        min_support_count = math.ceil(min_support * n_tx)
    else:
        min_support_count = int(min_support)

    tree, global_freq, n_tx = build_fptree(transactions, min_support_count=min_support_count)
    if tree is None:
        return {}, n_tx

    # cập nhật header_table: đã có [count, head], count đã là support trong cây
    # Khai thác
    freq_itemsets = {}
    mine_tree(tree, prefix=[], min_support_count=min_support_count, freq_itemsets=freq_itemsets)
    return freq_itemsets, n_tx

# =========================
# 5) Maximal itemsets
# =========================
def maximal_itemsets(freq_itemsets):
    """
    Lọc ra các maximal itemsets: không có superset nào khác cũng frequent.
    freq_itemsets: dict{frozenset: count}
    """
    by_len = sorted(freq_itemsets.keys(), key=lambda s: (-len(s), tuple(s)))
    maximals = []
    maximals_setlist = []  # để superset check nhanh
    for s in by_len:
        is_subset_of_existing_max = any(ms.issuperset(s) for ms in maximals_setlist)
        if not is_subset_of_existing_max:
            maximals.append((s, freq_itemsets[s]))
            maximals_setlist.append(s)
    return maximals

# =========================
# 6) Association rules
# =========================
def all_nonempty_proper_subsets(itemset):
    items = list(itemset)
    for r in range(1, len(items)):
        for A in combinations(items, r):
            yield frozenset(A)

def association_rules(freq_itemsets, n_tx, min_conf=0.6):
    """
    Sinh luật A -> B (A ⋂ B = Ø, A ∪ B là frequent)
    - confidence = supp(A∪B) / supp(A)
    - lift = conf / supp(B)
    - leverage = supp(A∪B) - supp(A)*supp(B)
    - conviction = (1 - supp(B)) / (1 - conf) (nếu conf < 1)
    """
    rules = []
    # Đổi sang support tỷ lệ
    supp = {k: v / n_tx for k, v in freq_itemsets.items()}
    for I, suppI in supp.items():
        if len(I) < 2:
            continue
        for A in all_nonempty_proper_subsets(I):
            B = I.difference(A)
            if not B:
                continue
            suppA = supp.get(A)
            suppB = supp.get(B)
            if suppA is None or suppB is None or suppA == 0:
                continue
            conf = suppI / suppA
            if conf + 1e-12 < min_conf:
                continue
            lift = conf / suppB if suppB > 0 else float("inf")
            leverage = suppI - suppA * suppB
            conviction = float("inf") if conf >= 1.0 else (1 - suppB) / (1 - conf)
            rules.append({
                "antecedent": tuple(sorted(A)),
                "consequent": tuple(sorted(B)),
                "support": suppI,
                "confidence": conf,
                "lift": lift,
                "leverage": leverage,
                "conviction": conviction
            })
    # sắp xếp theo confidence rồi lift
    rules.sort(key=lambda r: (-r["confidence"], -r["lift"], -r["support"]))
    return rules

# =========================
# 7) Helper: Lưu CSV
# =========================
def save_frequent_itemsets_csv(freq_itemsets, n_tx, path):
    import csv
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["itemset", "length", "support_count", "support"])
        for iset, cnt in sorted(freq_itemsets.items(), key=lambda kv: (len(kv[0]), kv[0])):
            w.writerow([" ".join(map(str, sorted(iset))), len(iset), cnt, cnt/n_tx])

def save_maximal_itemsets_csv(maximals, n_tx, path):
    import csv
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["itemset", "length", "support_count", "support"])
        for iset, cnt in sorted(maximals, key=lambda kv: (-len(kv[0]), kv[0])):
            w.writerow([" ".join(map(str, sorted(iset))), len(iset), cnt, cnt/n_tx])

def save_rules_csv(rules, path, top=None):
    import csv
    rows = rules if top is None else rules[:top]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["antecedent","consequent","support","confidence","lift","leverage","conviction"])
        for r in rows:
            w.writerow([
                " ".join(map(str, r["antecedent"])),
                " ".join(map(str, r["consequent"])),
                f'{r["support"]:.6f}',
                f'{r["confidence"]:.6f}',
                f'{r["lift"]:.6f}',
                f'{r["leverage"]:.6f}',
                f'{r["conviction"]:.6f}' if math.isfinite(r["conviction"]) else "inf",
            ])

# =========================
# 8) Main CLI
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="retail.dat",help="Đường dẫn file FIMI (retail.dat)")
    parser.add_argument("--min_support", type=float, default=0.02, help="Tối thiểu support (tỉ lệ, vd 0.02)")
    parser.add_argument("--min_conf", type=float, default=0.6, help="Tối thiểu confidence (vd 0.6)")
    parser.add_argument("--limit", type=int, default=None, help="Chỉ đọc N dòng đầu để test")
    parser.add_argument("--out_dir", type=str, default="./out", help="Thư mục lưu CSV")
    parser.add_argument("--top_rules", type=int, default=100, help="Số luật đầu để xem nhanh")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Đọc dữ liệu từ: {args.data}")
    tx = load_transactions(args.data, limit=args.limit)
    print(f"Số giao dịch đọc được: {len(tx):,}")

    print(f"Chạy FP-Growth với min_support={args.min_support} ...")
    freq_itemsets, n_tx = fpgrowth(tx, min_support=args.min_support)
    print(f"Số giao dịch hiệu dụng: {n_tx:,}")
    print(f"Số frequent itemsets: {len(freq_itemsets):,}")

    # Lưu frequent itemsets
    fi_path = os.path.join(args.out_dir, "frequent_itemsets.csv")
    save_frequent_itemsets_csv(freq_itemsets, n_tx, fi_path)
    print(f"Đã lưu frequent itemsets -> {fi_path}")

    # Maximal itemsets
    maximals = maximal_itemsets(freq_itemsets)
    print(f"Số maximal itemsets: {len(maximals):,}")
    mx_path = os.path.join(args.out_dir, "maximal_itemsets.csv")
    save_maximal_itemsets_csv(maximals, n_tx, mx_path)
    print(f"Đã lưu maximal itemsets -> {mx_path}")

    # Association rules
    print(f"Sinh association rules với min_conf={args.min_conf} ...")
    rules = association_rules(freq_itemsets, n_tx, min_conf=args.min_conf)
    print(f"Số rules: {len(rules):,}")
    ar_path = os.path.join(args.out_dir, "association_rules.csv")
    save_rules_csv(rules, ar_path)
    print(f"Đã lưu rules -> {ar_path}")

    # In nhanh top vài luật
    print("\nTop luật (theo confidence rồi lift):")
    for r in rules[:args.top_rules]:
        A = ", ".join(map(str, r["antecedent"]))
        B = ", ".join(map(str, r["consequent"]))
        print(f"{A} -> {B} | supp={r['support']:.4f}, conf={r['confidence']:.3f}, lift={r['lift']:.3f}")

if __name__ == "__main__":
    main()
