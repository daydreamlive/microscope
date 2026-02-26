#pragma once
#include <string>
#include <vector>
#include <map>
#include <set>
#include <cstdint>

class CLIPTokenizer {
public:
    static constexpr int BOS_TOKEN_ID = 49406;
    static constexpr int EOS_TOKEN_ID = 49407;
    static constexpr int PAD_TOKEN_ID = 49407;
    static constexpr int MAX_LENGTH   = 77;

    // Load from vocab.json + merges.txt files
    bool load(const std::string& vocab_path, const std::string& merges_path);

    // Tokenize text → padded vector of length 77
    std::vector<int32_t> tokenize(const std::string& text);

private:
    std::map<int, std::u32string> byte_encoder_;
    std::map<std::u32string, int> encoder_;
    std::map<std::pair<std::u32string, std::u32string>, int> bpe_ranks_;

    std::u32string bpe(const std::u32string& token);
    std::vector<int32_t> encode(const std::string& text);

    static std::vector<std::pair<int, std::u32string>> bytes_to_unicode();
    static std::set<std::pair<std::u32string, std::u32string>>
        get_pairs(const std::vector<std::u32string>& subwords);
    static std::u32string utf8_to_utf32(const std::string& s);
    static std::string utf32_to_utf8(const std::u32string& s);
    static std::vector<std::string> regex_split(const std::string& text);
};
