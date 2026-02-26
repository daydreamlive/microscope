#include "tokenizer.h"
#include "json.hpp"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <regex>
#include <set>
#include <codecvt>
#include <locale>
#include <cstdio>

using json = nlohmann::json;

// ── UTF conversion ──────────────────────────────────────────────────────────

std::u32string CLIPTokenizer::utf8_to_utf32(const std::string& s) {
    std::u32string result;
    size_t i = 0;
    while (i < s.size()) {
        uint32_t cp;
        unsigned char c = s[i];
        if (c < 0x80) {
            cp = c; i += 1;
        } else if (c < 0xE0) {
            cp = (c & 0x1F) << 6;
            if (i + 1 < s.size()) cp |= (s[i+1] & 0x3F);
            i += 2;
        } else if (c < 0xF0) {
            cp = (c & 0x0F) << 12;
            if (i + 1 < s.size()) cp |= (s[i+1] & 0x3F) << 6;
            if (i + 2 < s.size()) cp |= (s[i+2] & 0x3F);
            i += 3;
        } else {
            cp = (c & 0x07) << 18;
            if (i + 1 < s.size()) cp |= (s[i+1] & 0x3F) << 12;
            if (i + 2 < s.size()) cp |= (s[i+2] & 0x3F) << 6;
            if (i + 3 < s.size()) cp |= (s[i+3] & 0x3F);
            i += 4;
        }
        result.push_back(static_cast<char32_t>(cp));
    }
    return result;
}

std::string CLIPTokenizer::utf32_to_utf8(const std::u32string& s) {
    std::string result;
    for (char32_t cp : s) {
        if (cp < 0x80) {
            result.push_back(static_cast<char>(cp));
        } else if (cp < 0x800) {
            result.push_back(static_cast<char>(0xC0 | (cp >> 6)));
            result.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        } else if (cp < 0x10000) {
            result.push_back(static_cast<char>(0xE0 | (cp >> 12)));
            result.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
            result.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        } else {
            result.push_back(static_cast<char>(0xF0 | (cp >> 18)));
            result.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
            result.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
            result.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        }
    }
    return result;
}

// ── Byte ↔ Unicode mapping (CLIP convention) ────────────────────────────────

std::vector<std::pair<int, std::u32string>> CLIPTokenizer::bytes_to_unicode() {
    std::vector<std::pair<int, std::u32string>> pairs;
    std::set<int> byte_set;
    // Printable ASCII
    for (int b = '!'; b <= '~'; ++b) {
        byte_set.insert(b);
        pairs.emplace_back(b, std::u32string(1, static_cast<char32_t>(b)));
    }
    // Latin supplement ranges
    for (int b = 161; b <= 172; ++b) {
        byte_set.insert(b);
        pairs.emplace_back(b, std::u32string(1, static_cast<char32_t>(b)));
    }
    for (int b = 174; b <= 255; ++b) {
        byte_set.insert(b);
        pairs.emplace_back(b, std::u32string(1, static_cast<char32_t>(b)));
    }
    // Map remaining bytes to 256+
    int n = 0;
    for (int b = 0; b < 256; ++b) {
        if (byte_set.find(b) == byte_set.end()) {
            pairs.emplace_back(b, std::u32string(1, static_cast<char32_t>(256 + n)));
            ++n;
        }
    }
    return pairs;
}

// ── Load vocab + merges ─────────────────────────────────────────────────────

bool CLIPTokenizer::load(const std::string& vocab_path, const std::string& merges_path) {
    // Load vocab.json → encoder map (token_string → id)
    std::ifstream vf(vocab_path);
    if (!vf.is_open()) {
        fprintf(stderr, "Cannot open vocab: %s\n", vocab_path.c_str());
        return false;
    }
    json vocab = json::parse(vf);
    for (auto& [key, val] : vocab.items()) {
        encoder_[utf8_to_utf32(key)] = val.get<int>();
    }

    // Load merges.txt → bpe_ranks
    std::ifstream mf(merges_path);
    if (!mf.is_open()) {
        fprintf(stderr, "Cannot open merges: %s\n", merges_path.c_str());
        return false;
    }
    std::string line;
    std::getline(mf, line); // skip header "#version: ..."
    int rank = 0;
    while (std::getline(mf, line)) {
        if (line.empty()) continue;
        size_t sp = line.find(' ');
        if (sp == std::string::npos) continue;
        auto first = utf8_to_utf32(line.substr(0, sp));
        auto second = utf8_to_utf32(line.substr(sp + 1));
        bpe_ranks_[{first, second}] = rank++;
    }

    // Build byte_encoder
    auto bu = bytes_to_unicode();
    for (auto& [b, u] : bu) {
        byte_encoder_[b] = u;
    }

    fprintf(stderr, "Tokenizer: %zu vocab entries, %d merges\n",
            encoder_.size(), rank);
    return true;
}

// ── BPE algorithm ───────────────────────────────────────────────────────────

std::set<std::pair<std::u32string, std::u32string>>
CLIPTokenizer::get_pairs(const std::vector<std::u32string>& subwords) {
    std::set<std::pair<std::u32string, std::u32string>> pairs;
    for (size_t i = 1; i < subwords.size(); ++i) {
        pairs.insert({subwords[i-1], subwords[i]});
    }
    return pairs;
}

std::u32string CLIPTokenizer::bpe(const std::u32string& token) {
    std::vector<std::u32string> word;
    for (size_t i = 0; i < token.size() - 1; ++i) {
        word.emplace_back(1, token[i]);
    }
    word.push_back(token.substr(token.size() - 1) + utf8_to_utf32("</w>"));

    auto pairs = get_pairs(word);
    if (pairs.empty()) {
        return token + utf8_to_utf32("</w>");
    }

    while (true) {
        auto min_pair = std::min_element(pairs.begin(), pairs.end(),
            [&](const auto& a, const auto& b) {
                bool a_found = bpe_ranks_.count(a);
                bool b_found = bpe_ranks_.count(b);
                if (!a_found) return false;
                if (!b_found) return true;
                return bpe_ranks_.at(a) < bpe_ranks_.at(b);
            });

        if (bpe_ranks_.find(*min_pair) == bpe_ranks_.end()) break;

        auto first = min_pair->first;
        auto second = min_pair->second;
        std::vector<std::u32string> new_word;
        size_t i = 0;

        while (i < word.size()) {
            auto it = std::find(word.begin() + i, word.end(), first);
            if (it == word.end()) {
                new_word.insert(new_word.end(), word.begin() + i, word.end());
                break;
            }
            new_word.insert(new_word.end(), word.begin() + i, it);
            i = static_cast<size_t>(std::distance(word.begin(), it));

            if (word[i] == first && i + 1 < word.size() && word[i + 1] == second) {
                new_word.push_back(first + second);
                i += 2;
            } else {
                new_word.push_back(word[i]);
                i += 1;
            }
        }

        word = new_word;
        if (word.size() == 1) break;
        pairs = get_pairs(word);
    }

    std::u32string result;
    for (size_t i = 0; i < word.size(); ++i) {
        result += word[i];
        if (i != word.size() - 1) result += U' ';
    }
    return result;
}

// ── Regex split (CLIP pattern) ──────────────────────────────────────────────

std::vector<std::string> CLIPTokenizer::regex_split(const std::string& text) {
    std::regex pat(R"('s|'t|'re|'ve|'m|'ll|'d|[a-zA-Z]+|[0-9]|[^ a-zA-Z0-9]+)",
                   std::regex::icase);
    std::vector<std::string> result;
    std::sregex_iterator it(text.begin(), text.end(), pat);
    std::sregex_iterator end;
    for (; it != end; ++it) {
        result.push_back(it->str());
    }
    return result;
}

// ── Encode text → token ids (without BOS/EOS) ──────────────────────────────

std::vector<int32_t> CLIPTokenizer::encode(const std::string& text) {
    // Whitespace clean + lowercase
    std::string cleaned = std::regex_replace(text, std::regex(R"(\s+)"), " ");
    // Strip
    auto start = cleaned.find_first_not_of(" \t\n\r");
    auto end = cleaned.find_last_not_of(" \t\n\r");
    if (start == std::string::npos) cleaned = "";
    else cleaned = cleaned.substr(start, end - start + 1);
    // Lowercase
    std::transform(cleaned.begin(), cleaned.end(), cleaned.begin(),
                   [](unsigned char c) { return std::tolower(c); });

    std::vector<int32_t> tokens;
    auto words = regex_split(cleaned);

    for (auto& word : words) {
        // Convert each byte through byte_encoder
        std::u32string utf32_token;
        for (unsigned char b : word) {
            utf32_token += byte_encoder_[b];
        }

        // Apply BPE
        auto bpe_result = bpe(utf32_token);

        // Split on spaces and look up each subword
        std::u32string::size_type pos = 0, sp;
        while ((sp = bpe_result.find(U' ', pos)) != std::u32string::npos) {
            auto sub = bpe_result.substr(pos, sp - pos);
            auto it = encoder_.find(sub);
            if (it != encoder_.end()) {
                tokens.push_back(it->second);
            }
            pos = sp + 1;
        }
        auto sub = bpe_result.substr(pos);
        auto it = encoder_.find(sub);
        if (it != encoder_.end()) {
            tokens.push_back(it->second);
        }
    }
    return tokens;
}

// ── Tokenize with BOS/EOS/padding to length 77 ─────────────────────────────

std::vector<int32_t> CLIPTokenizer::tokenize(const std::string& text) {
    auto tokens = encode(text);

    // Insert BOS at start
    tokens.insert(tokens.begin(), BOS_TOKEN_ID);

    // Truncate or pad to MAX_LENGTH
    if (tokens.size() > MAX_LENGTH - 1) {
        tokens.resize(MAX_LENGTH - 1);
        tokens.push_back(EOS_TOKEN_ID);
    } else {
        tokens.push_back(EOS_TOKEN_ID);
        tokens.resize(MAX_LENGTH, PAD_TOKEN_ID);
    }

    return tokens;
}
