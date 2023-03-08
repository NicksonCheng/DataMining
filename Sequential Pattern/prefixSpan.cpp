#include <algorithm>
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <set>
#include <sstream>
#include <vector>
using namespace std;
typedef set<int> Itemset;
typedef vector<set<int>> Sequence;
typedef vector<vector<set<int>>> Dataset;
typedef map<Sequence, int> SeqeunceCount;
template <typename T>
void slice(std::vector<T> &v, int m, int n) {
    int k = n - m + 1;

    auto it = v.cbegin() + m;
    while (it != v.cend() && k--) {
        it = v.erase(it);
    }
}
int access_set(Itemset items, int idx) {
    auto it = items.begin();
    advance(it, idx);
    return *it;
}

void printDataset(Dataset data) {
    for (int i = 0; i < data.size(); ++i) {
        cout << "id:" << i + 1 << " ";
        for (int j = 0; j < data[i].size(); ++j) {
            cout << "(";
            Itemset itemset = data[i][j];
            for (auto it = itemset.begin(); it != itemset.end(); ++it) {
                cout << *it;
                if (next(it) != itemset.end()) cout << " ";
            }
            cout << ")";
        }
        cout << endl;
    }
}
void printCandidate(Sequence seq, int c, string type) {
    cout << type;
    for (int j = 0; j < seq.size(); ++j) {
        cout << "(";
        Itemset itemset = seq[j];
        for (auto it = itemset.begin(); it != itemset.end(); ++it) {
            cout << *it;
            if (next(it) != itemset.end()) cout << " ";
        }
        cout << ")";
    }
    cout << "->" << c << endl;
}
void printSupport(SeqeunceCount seq_count) {
    for (auto it = seq_count.begin(); it != seq_count.end(); ++it) {
        Sequence seq = it->first;
        for (auto it2 = seq.begin(); it2 != seq.end(); ++it2) {
            cout << "(";
            Itemset item = *it2;
            for (auto &it3 : item)
                cout << it3 << " ";
            cout << ")";
        }
        cout << "-> support=" << it->second << endl;
    }
}
int countSubsequence(Sequence candidate_seq, Sequence trans_seq, int candidate_idx, int trans_idx) {
    // subsequnce
    if (candidate_idx >= candidate_seq.size())
        return 1;
    Itemset target_itemset = candidate_seq[candidate_idx];
    int count = 0;
    for (int i = trans_idx; i < trans_seq.size(); ++i) {
        // need to check whether target_itemset is subset of trans_seq[i] itemset
        Itemset trans_itemsets = trans_seq[i];
        if (includes(trans_itemsets.begin(), trans_itemsets.end(), target_itemset.begin(), target_itemset.end())) {
            count += countSubsequence(candidate_seq, trans_seq, candidate_idx + 1, i + 1);
        }
    }
    return count;
}
SeqeunceCount generateCandidate(SeqeunceCount sequence_count, Dataset data, set<int> one_freq_items, int min_freq) {
    SeqeunceCount candidate_seq_count;
    for (auto it = sequence_count.begin(); it != sequence_count.end(); ++it) {
        Sequence candidate_seq(it->first.begin(), it->first.end());
        for (auto &it2 : one_freq_items) {
            // expand the sequence set
            // generate sequence set <(1)> -> <(1)(2)> <(1)(3)> ....
            Itemset itemset = {it2};
            candidate_seq.push_back(itemset);

            int c = 0;
            for (int i = 0; i < data.size(); ++i)
                c += countSubsequence(candidate_seq, data[i], 0, 0);
            // printCandidate(candidate_seq, c, "sequence set candidate:");
            if (c >= min_freq)
                candidate_seq_count[candidate_seq] = c;
            candidate_seq.pop_back();
        }
        candidate_seq = it->first;
        Itemset last_itemset = candidate_seq.back();
        candidate_seq.pop_back();
        for (auto &it2 : one_freq_items) {
            // expand the itemset in sequence
            // generate itemset  <(1)> -> <(1,2)> <(1,3)> ....
            if (last_itemset.find(it2) != last_itemset.end())
                continue;
            last_itemset.insert(it2);
            candidate_seq.push_back(last_itemset);
            int c = 0;
            for (int i = 0; i < data.size(); ++i)
                c += countSubsequence(candidate_seq, data[i], 0, 0);
            // printCandidate(candidate_seq, c, "item set candidate:");
            if (c >= min_freq)
                candidate_seq_count[candidate_seq] = c;
            // back to original sequence
            candidate_seq.pop_back();
            candidate_seq.push_back(last_itemset);
        }
    }
    return candidate_seq_count;
}
int main(int argc, char const *argv[]) {
    Dataset dataset;
    SeqeunceCount seq_count;
    set<int> one_freq_items;
    ifstream infile;
    ofstream outfile;
    infile.open("seqdata.txt");
    outfile.open("output.txt");
    string line;
    double min_sup = 0.025;
    while (getline(infile, line)) {
        /* code */
        Sequence sequnece;
        Itemset itemset;
        int seq_id = 0;
        int item_id = 0;
        int item = 0;
        int curr_id = 0;

        stringstream ss(line);
        ss >> seq_id;
        ss >> curr_id >> item;
        itemset.insert(item);
        one_freq_items.insert(item);
        // craete database
        while (ss >> item_id >> item) {
            /* code */
            if (item_id != curr_id) {
                sequnece.push_back(itemset);
                itemset.clear();

                curr_id = item_id;
            }
            // new item id item
            itemset.insert(item);
            one_freq_items.insert(item);
        }
        // in loop end, there still have item in itemset
        // only one itemsets
        sequnece.push_back(itemset);
        dataset.push_back(sequnece);
    }
    int min_freq = (int)(min_sup * dataset.size());

    // printDataset(dataset);
    /* find 1 frequent items */
    for (int i = 0; i < dataset.size(); ++i) {
        for (int j = 0; j < dataset[i].size(); ++j) {
            Itemset items = dataset[i][j];
            // seperate the items and use the value to create 1-element itemset
            for (auto &item : items) {
                Itemset new_item = {item};
                Sequence seq;
                seq.push_back(new_item);
                if (seq_count.find(seq) == seq_count.end()) {
                    seq_count[seq] = 1;
                } else
                    seq_count[seq]++;
            }
        }
    }
    cout << "min_freq=" << min_freq << endl;
    for (auto it = one_freq_items.begin(); it != one_freq_items.end();) {
        Sequence one_seq = {{*it}};
        if (seq_count[one_seq] < min_freq) {
            seq_count.erase(one_seq);
            auto erase_it = it++;
            one_freq_items.erase(*erase_it);
        } else {
            ++it;
        }
    }

    while (!seq_count.empty()) {
        /* code */
        auto it = seq_count.begin();
        int seq_num = it->first.size();
        cout << seq_num << " "
             << "frequent seqence:" << seq_count.size() << endl;
        printSupport(seq_count);
        seq_count = generateCandidate(seq_count, dataset, one_freq_items, min_freq);
    }

    infile.close();
    outfile.close();
    return 0;
}
