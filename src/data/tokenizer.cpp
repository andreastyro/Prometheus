#include "ml/data/tokenizer.hpp"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <stdexcept>
#include <cctype>

using namespace std;

// ── Private helpers ────────────────────────────────────────────────────────────

void Tokenizer::add_special_tokens() {
    id_to_token_ = {"<UNK>", "<PAD>", "<BOS>", "<EOS>"};
    token_to_id_.clear();
    for (int i = 0; i < (int)id_to_token_.size(); i++)
        token_to_id_[id_to_token_[i]] = i;
}

vector<string> Tokenizer::split(const string& text) const {
    vector<string> tokens;
    string cur;
    for (char c : text) {
        if (isspace((unsigned char)c)) {
            if (!cur.empty()) { tokens.push_back(cur); cur.clear(); }
        } else {
            cur += (char)tolower((unsigned char)c);
        }
    }
    if (!cur.empty()) tokens.push_back(cur);
    return tokens;
}

void Tokenizer::build_vocab(const unordered_map<string, int>& freq, int max_vocab) {
    // Sort by frequency descending, then alphabetically for ties (determinism)
    vector<pair<int, string>> ranked;
    ranked.reserve(freq.size());
    for (auto& [tok, cnt] : freq)
        ranked.push_back({cnt, tok});
    sort(ranked.begin(), ranked.end(), [](const pair<int,string>& a, const pair<int,string>& b) {
        return a.first != b.first ? a.first > b.first : a.second < b.second;
    });

    // Reserve space for specials (already in id_to_token_) then add words
    int remaining = max_vocab - (int)id_to_token_.size();
    for (auto& [cnt, tok] : ranked) {
        if (remaining-- <= 0) break;
        if (token_to_id_.count(tok)) continue; // skip if special token name collides
        token_to_id_[tok] = (int)id_to_token_.size();
        id_to_token_.push_back(tok);
    }
}

// ── Public API ─────────────────────────────────────────────────────────────────

void Tokenizer::build_from_text(const string& text, int max_vocab) {
    add_special_tokens();
    auto tokens = split(text);
    unordered_map<string, int> freq;
    for (auto& t : tokens) freq[t]++;
    build_vocab(freq, max_vocab);
}

void Tokenizer::build_from_file(const string& path, int max_vocab) {
    ifstream file(path);
    if (!file) throw runtime_error("Tokenizer::build_from_file: cannot open " + path);

    add_special_tokens();
    unordered_map<string, int> freq;
    string line;
    while (getline(file, line)) {
        for (auto& t : split(line))
            freq[t]++;
    }
    build_vocab(freq, max_vocab);
}

void Tokenizer::save(const string& path) const {
    ofstream file(path);
    if (!file) throw runtime_error("Tokenizer::save: cannot open " + path);
    for (auto& tok : id_to_token_)
        file << tok << "\n";
}

void Tokenizer::load(const string& path) {
    ifstream file(path);
    if (!file) throw runtime_error("Tokenizer::load: cannot open " + path);

    token_to_id_.clear();
    id_to_token_.clear();
    string line;
    while (getline(file, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        token_to_id_[line] = (int)id_to_token_.size();
        id_to_token_.push_back(line);
    }
}

vector<int> Tokenizer::encode(const string& text, bool add_bos, bool add_eos) const {
    vector<int> ids;
    if (add_bos) ids.push_back(BOS_ID);
    for (auto& tok : split(text)) {
        auto it = token_to_id_.find(tok);
        ids.push_back(it != token_to_id_.end() ? it->second : UNK_ID);
    }
    if (add_eos) ids.push_back(EOS_ID);
    return ids;
}

string Tokenizer::decode(const vector<int>& ids, bool skip_special) const {
    string out;
    for (int id : ids) {
        if (id < 0 || id >= (int)id_to_token_.size()) continue;
        if (skip_special && id < 4) continue; // skip UNK/PAD/BOS/EOS
        if (!out.empty()) out += ' ';
        out += id_to_token_[id];
    }
    return out;
}

int Tokenizer::token_to_id(const string& tok) const {
    auto it = token_to_id_.find(tok);
    return it != token_to_id_.end() ? it->second : UNK_ID;
}

const string& Tokenizer::id_to_token(int id) const {
    static const string unknown = "<UNK>";
    if (id < 0 || id >= (int)id_to_token_.size()) return unknown;
    return id_to_token_[id];
}
