#pragma once
#include <string>
#include <vector>
#include <unordered_map>

// Word-level tokenizer.
//
// Splits text on whitespace, lowercases, then maps tokens to integer IDs.
// The vocabulary is built from a frequency count; only the top max_vocab
// most frequent tokens are kept. Rare tokens map to <UNK>.
//
// Special token IDs are fixed:
//   0 — <UNK>   unknown word
//   1 — <PAD>   padding (for batch alignment)
//   2 — <BOS>   beginning of sequence
//   3 — <EOS>   end of sequence
//
// Usage:
//   Tokenizer tok;
//   tok.build_from_file("wikitext-103/wiki.train.tokens");
//   tok.save("vocab.txt");
//   auto ids = tok.encode("the quick brown fox", true, true);  // add BOS/EOS
//   std::string text = tok.decode(ids);
class Tokenizer {
public:
    static constexpr int UNK_ID = 0;
    static constexpr int PAD_ID = 1;
    static constexpr int BOS_ID = 2;
    static constexpr int EOS_ID = 3;

    Tokenizer() { add_special_tokens(); }

    // Build vocabulary from a text file (one line at a time).
    void build_from_file(const std::string& path, int max_vocab = 50000);

    // Build vocabulary from a string (useful for small corpora).
    void build_from_text(const std::string& text, int max_vocab = 50000);

    // Persist vocabulary: one token per line, in ID order.
    void save(const std::string& path) const;
    void load(const std::string& path);

    // Map a string to a sequence of token IDs.
    std::vector<int> encode(const std::string& text,
                            bool add_bos = false,
                            bool add_eos = false) const;

    // Map token IDs back to a string.
    // If skip_special=true, <UNK>/<PAD>/<BOS>/<EOS> are omitted.
    std::string decode(const std::vector<int>& ids, bool skip_special = true) const;

    int  vocab_size() const { return static_cast<int>(id_to_token_.size()); }
    bool has_token(const std::string& tok) const { return token_to_id_.count(tok) > 0; }
    int  token_to_id(const std::string& tok) const;
    const std::string& id_to_token(int id) const;

private:
    std::unordered_map<std::string, int> token_to_id_;
    std::vector<std::string>             id_to_token_;

    void add_special_tokens();
    std::vector<std::string> split(const std::string& text) const;
    void build_vocab(const std::unordered_map<std::string, int>& freq, int max_vocab);
};
