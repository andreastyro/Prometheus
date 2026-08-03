#include "ml/data/tokenizer.hpp"
#include <cassert>
#include <cstdio>

int main() {
    Tokenizer tok;

    // Build from string
    std::string corpus = "the cat sat on the mat the cat is happy";
    tok.build_from_text(corpus, 100);

    assert(tok.vocab_size() >= 4 && "must have at least the 4 special tokens");
    assert(tok.has_token("the")    && "frequent word must be in vocab");
    assert(tok.has_token("cat")    && "'cat' must be in vocab");
    assert(!tok.has_token("xyz")   && "unknown word must not be in vocab");

    // Encode / decode round-trip
    auto ids = tok.encode("the cat sat", false, false);
    assert(ids.size() == 3);
    std::string decoded = tok.decode(ids);
    assert(decoded == "the cat sat");

    // BOS / EOS
    auto ids_bos = tok.encode("cat", true, true);
    assert(ids_bos.front() == Tokenizer::BOS_ID);
    assert(ids_bos.back()  == Tokenizer::EOS_ID);

    // Unknown word maps to UNK
    auto ids_unk = tok.encode("notaword", false, false);
    assert(ids_unk[0] == Tokenizer::UNK_ID);

    // Save / load
    tok.save("_test_vocab.txt");
    Tokenizer tok2;
    tok2.load("_test_vocab.txt");
    assert(tok2.vocab_size() == tok.vocab_size());
    assert(tok2.encode("the cat", false, false) == tok.encode("the cat", false, false));

    printf("test_tokenizer PASSED\n");
    return 0;
}
