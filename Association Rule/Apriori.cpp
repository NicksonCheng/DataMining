#include <algorithm>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <vector>
using namespace std;
typedef vector<set<int>> Dataset;
typedef set<int> Itemset;
typedef map<Itemset, int> ItemsetCount;
ofstream outfile;
void printDataset(Dataset data) {
    for (int i = 0; i < data.size(); ++i) {
        Itemset transactions = data[i];
        for (auto &t : transactions) {
            cout << t << " ";
        }
        cout << endl;
    }
}
void removeFromDataBase(Itemset delete_set, Dataset &data) {
    for (int i = 0; i < data.size(); ++i) {
        Itemset transactions = data[i];
        Itemset result;
        set_difference(transactions.begin(), transactions.end(),
                       delete_set.begin(), delete_set.end(), inserter(result, result.begin()));
        data[i] = result;
    }
}
void findMiniConfidence(ItemsetCount total_itemset_count, Dataset data, double min_conf = 0.66) {
    for (auto it1 = total_itemset_count.begin(); it1 != total_itemset_count.end(); ++it1) {
        Itemset non_repeat_itemset;
        for (auto it2 = total_itemset_count.begin(); it2 != total_itemset_count.end(); ++it2) {
            if (it1 == it2) continue;
            Itemset f1 = it1->first;
            Itemset f2 = it2->first;
            Itemset candidate(f1);
            candidate.insert(f2.begin(), f2.end());
            if (non_repeat_itemset == candidate) {
                break;
            }
            // this candidate is in frequent items and is larger than min_conf
            auto cand_it = total_itemset_count.find(candidate);
            // cout << "successful" << endl;
            if (cand_it != total_itemset_count.end()) {
                // f1 is subset of f2 is not legal(must large than min_conf)
                if (includes(f1.begin(), f1.end(), f2.begin(), f2.end())) {
                    continue;
                }
                if ((double)cand_it->second / it1->second > min_conf) {
                    // avoid repeat output association rule

                    non_repeat_itemset = candidate;
                    // output this association rule

                    cout << "{";
                    outfile << "{";
                    for (auto &i : f1) {
                        cout << i << ",";
                        outfile << i << ",";
                    }

                    cout << "} -> {";
                    outfile << "} -> {";
                    for (auto &i : f2) {
                        cout << i << ",";
                        outfile << i << ",";
                    }
                    cout << "} (" << cand_it->second << "/" << it1->second << ")" << endl;
                    outfile << "} (" << cand_it->second << "/" << it1->second << ")" << endl;
                }
            }
        }
    }
}
ItemsetCount generateCandidate(ItemsetCount itemset_count, Dataset &data, double min_freq) {
    // generate Ck+1 from Lk
    ItemsetCount candidate_count;
    for (auto it1 = itemset_count.begin(); it1 != itemset_count.end(); ++it1) {
        for (auto it2 = next(it1); it2 != itemset_count.end(); ++it2) {
            Itemset candidate_itemset(it1->first.begin(), it1->first.end());
            candidate_itemset.insert(it2->first.begin(), it2->first.end());
            candidate_count[candidate_itemset] = 0;
            // check is frequent itemsets
            for (int j = 0; j < data.size(); ++j) {
                Itemset transactions = data[j];
                if (includes(transactions.begin(), transactions.end(), candidate_itemset.begin(), candidate_itemset.end())) {
                    candidate_count[candidate_itemset]++;
                }
            }
            if (candidate_count[candidate_itemset] < min_freq) {
                // removeFromDataBase(candidate_itemset, data);
                candidate_count.erase(candidate_itemset);
            }
            // prune(candidate_count, data, min_sup);
        }
    }
    return candidate_count;
}
int main(int argc, char const *argv[]) {
    /* code */
    Dataset dataset;
    ItemsetCount itemset_count;
    ifstream infile("input.txt");
    outfile.open("output.txt");
    string line;
    int total_transactions = 0;
    int set_num = 1;
    double min_sup = 0.05;
    double min_conf = 0.66;
    cout << "input min_sup(default=0.05):";
    cin >> min_sup;
    cout << "input min_conf(default=0.66)";
    cin >> min_conf;
    while (getline(infile, line)) {
        // cout << line << endl;
        int item;
        Itemset itemset;
        stringstream ss(line);

        while (ss >> item) {
            /* code */
            itemset.insert(item);
        }
        dataset.push_back(itemset);
        total_transactions++;
    }
    double min_freq = total_transactions * min_sup;
    // Ck: Candidate itemset of size k
    // Lk : frequent itemset of size k
    //  find frequent item set with L1

    for (Itemset itemset : dataset) {
        // cout << itemset.size() << endl;
        for (auto it = itemset.begin(); it != itemset.end(); ++it) {
            Itemset item = {*it};
            if (itemset_count.find(item) == itemset_count.end()) {
                // can't find any item in itemset_count
                itemset_count[item] = 1;
            } else {
                itemset_count[item]++;
            }
        }
    }

    // prune the itemset less than minimum support
    for (auto it = itemset_count.begin(); it != itemset_count.end();) {
        // cout << "frequent:" << it->second << endl;
        if (it->second < min_freq) {
            // remove it from database
            // removeFromDataBase(it->first, dataset);
            auto erase_it = it++;
            itemset_count.erase(erase_it);
        } else {
            ++it;
        }
    }
    ItemsetCount total_itemset_count = itemset_count;
    cout << "minimum support=" << min_sup << " minimum confidence=" << min_conf << endl;
    outfile << "minimum support=" << min_sup << " minimum confidence=" << min_conf << endl;
    while (!itemset_count.empty()) {
        /* code */
        // find Lk frequent itemset
        // cout << "current database" << endl;
        // printDataset(dataset);
        cout << "----------Frequent Item Set " << set_num << "----------" << endl;
        outfile << "----------Frequent Item Set " << set_num++ << "----------" << endl;
        for (auto it = itemset_count.begin(); it != itemset_count.end(); ++it) {
            Itemset item = it->first;
            cout << "{";
            outfile << "{";
            for (auto i : item) {
                cout << i << ",";
                outfile << i << ",";
            }
            cout << "}";
            outfile << "}";

            cout << ":" << it->second << endl;
            outfile << ":" << it->second << endl;
        }
        itemset_count = generateCandidate(itemset_count, dataset, min_freq);
        total_itemset_count.insert(itemset_count.begin(), itemset_count.end());
    }
    cout << "----------Minimum Confidence----------" << endl;
    outfile << "----------Minimum Confidence----------" << endl;
    findMiniConfidence(total_itemset_count, dataset, min_conf);
    outfile.close();
    return 0;
}
