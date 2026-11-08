#include "ml/utils/model_io.hpp"
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>

using namespace std;

namespace {

// safetensors is little-endian; so is every platform this builds for
static_assert(sizeof(float) == 4 && sizeof(double) == 8, "unexpected float sizes");
inline bool little_endian() { const uint16_t x = 1; return *reinterpret_cast<const uint8_t*>(&x) == 1; }

const char* st_name(DType d) {
    switch (d) {
        case DType::Float32:  return "F32";
        case DType::Float64:  return "F64";
        case DType::Float16:  return "F16";
        case DType::BFloat16: return "BF16";
        case DType::Int64:    return "I64";
        case DType::Int32:    return "I32";
        case DType::Bool:     return "BOOL";
    }
    return "?";
}

DType st_dtype(const string& s, const string& path) {
    if (s == "F32")  return DType::Float32;
    if (s == "F64")  return DType::Float64;
    if (s == "F16")  return DType::Float16;
    if (s == "BF16") return DType::BFloat16;
    if (s == "I64")  return DType::Int64;
    if (s == "I32")  return DType::Int32;
    if (s == "BOOL") return DType::Bool;
    throw runtime_error("load_safetensors: '" + path + "' has a tensor of dtype " + s +
                        ", which this library does not support");
}

string json_escape(const string& s) {
    string out;
    for (unsigned char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if (c < 0x20) { char buf[8]; snprintf(buf, sizeof buf, "\\u%04x", c); out += buf; }
                else out += (char)c;
        }
    }
    return out;
}

// ── a strict reader for the JSON a safetensors header contains ───────────────
// Objects, arrays, strings, non-negative integers. Anything else is an error:
// the header comes from a file, and a lenient parser is how malformed input
// turns into an out-of-range read.

struct Json {
    enum Kind { Object, Array, String, Number } kind = Number;
    vector<pair<string, Json>> obj;
    vector<Json> arr;
    string str;
    uint64_t num = 0;

    const Json* get(const string& key) const {
        for (auto& kv : obj) if (kv.first == key) return &kv.second;
        return nullptr;
    }
};

class Parser {
public:
    Parser(const string& text, const string& path) : s_(text), path_(path) {}
    Json parse() {
        Json v = value();
        skip_ws();
        if (i_ != s_.size()) fail("trailing characters");
        return v;
    }
private:
    const string& s_;
    const string& path_;
    size_t i_ = 0;
    int depth_ = 0;

    [[noreturn]] void fail(const string& why) {
        throw runtime_error("load_safetensors: '" + path_ + "' has a malformed header (" + why +
                            " at byte " + to_string(i_) + ")");
    }
    void skip_ws() { while (i_ < s_.size() && (s_[i_] == ' ' || s_[i_] == '\n' || s_[i_] == '\r' || s_[i_] == '\t')) i_++; }
    char peek() { skip_ws(); if (i_ >= s_.size()) fail("unexpected end"); return s_[i_]; }
    void expect(char c) { if (peek() != c) fail(string("expected '") + c + "'"); i_++; }

    Json value() {
        if (++depth_ > 8) fail("nesting too deep");
        Json v;
        const char c = peek();
        if (c == '{') {
            v.kind = Json::Object; i_++;
            if (peek() == '}') { i_++; }
            else for (;;) {
                string k = str();
                expect(':');
                v.obj.emplace_back(std::move(k), value());
                if (peek() == ',') { i_++; continue; }
                expect('}'); break;
            }
        } else if (c == '[') {
            v.kind = Json::Array; i_++;
            if (peek() == ']') { i_++; }
            else for (;;) {
                v.arr.push_back(value());
                if (peek() == ',') { i_++; continue; }
                expect(']'); break;
            }
        } else if (c == '"') {
            v.kind = Json::String; v.str = str();
        } else if (c >= '0' && c <= '9') {
            v.kind = Json::Number;
            uint64_t n = 0;
            while (i_ < s_.size() && s_[i_] >= '0' && s_[i_] <= '9') {
                const uint64_t d = (uint64_t)(s_[i_++] - '0');
                if (n > (numeric_limits<uint64_t>::max() - d) / 10) fail("number too large");
                n = n * 10 + d;
            }
            v.num = n;
        } else {
            fail("unexpected character");
        }
        depth_--;
        return v;
    }

    string str() {
        expect('"');
        string out;
        while (i_ < s_.size() && s_[i_] != '"') {
            char c = s_[i_++];
            if (c != '\\') { out += c; continue; }
            if (i_ >= s_.size()) fail("unterminated escape");
            const char e = s_[i_++];
            switch (e) {
                case '"': out += '"'; break;   case '\\': out += '\\'; break;
                case '/': out += '/'; break;   case 'b': out += '\b'; break;
                case 'f': out += '\f'; break;  case 'n': out += '\n'; break;
                case 'r': out += '\r'; break;  case 't': out += '\t'; break;
                case 'u': {
                    if (i_ + 4 > s_.size()) fail("short \\u escape");
                    const unsigned cp = (unsigned)stoul(s_.substr(i_, 4), nullptr, 16);
                    i_ += 4;
                    // UTF-8 for the basic plane; names are almost always ASCII
                    if (cp < 0x80) out += (char)cp;
                    else if (cp < 0x800) { out += (char)(0xC0 | (cp >> 6)); out += (char)(0x80 | (cp & 0x3F)); }
                    else { out += (char)(0xE0 | (cp >> 12)); out += (char)(0x80 | ((cp >> 6) & 0x3F));
                           out += (char)(0x80 | (cp & 0x3F)); }
                    break;
                }
                default: fail("bad escape");
            }
        }
        if (i_ >= s_.size()) fail("unterminated string");
        i_++;
        return out;
    }
};

constexpr uint64_t MAX_HEADER = 100ull << 20;   // the spec's limit: 100 MB

bool is_safetensors(const string& path) {
    ifstream f(path, ios::binary | ios::ate);
    if (!f) return false;
    const uint64_t size = (uint64_t)f.tellg();
    if (size < 10) return false;
    f.seekg(0);
    uint64_t n = 0;
    f.read(reinterpret_cast<char*>(&n), 8);
    char first = 0;
    f.read(&first, 1);
    return f && n >= 2 && n <= MAX_HEADER && n <= size - 8 && first == '{';
}

// ── the previous formats, read for compatibility ─────────────────────────────

// [count] then per tensor [ndims][dims as int32][float32 values]
vector<TensorPtr> read_legacy_tensors(ifstream& file, const string& path, int count) {
    if (count < 0 || count > 1000000)
        throw runtime_error("load: '" + path + "' is not a valid model file");
    vector<TensorPtr> params;
    for (int i = 0; i < count; i++) {
        int ndims = 0;
        file.read(reinterpret_cast<char*>(&ndims), sizeof(int));
        if (!file || ndims < 0 || ndims > 16)
            throw runtime_error("load: '" + path + "' is truncated or corrupt");
        Shape shape(ndims);
        for (int d = 0; d < ndims; d++) {
            int dim = 0;
            file.read(reinterpret_cast<char*>(&dim), sizeof(int));
            if (!file || dim < 0) throw runtime_error("load: '" + path + "' is truncated or corrupt");
            shape[d] = dim;
        }
        auto t = make_shared<Tensor>(shape);
        file.read(reinterpret_cast<char*>(t->data()), t->numel() * sizeof(float));
        if (!file) throw runtime_error("load: '" + path + "' is truncated");
        params.push_back(t);
    }
    return params;
}

vector<TensorPtr> in_order(const NamedTensors& named) {
    vector<TensorPtr> out;
    for (auto& kv : named) out.push_back(kv.second);
    return out;
}

NamedTensors numbered(const vector<TensorPtr>& params) {
    NamedTensors named;
    char buf[16];
    for (size_t i = 0; i < params.size(); i++) {
        snprintf(buf, sizeof buf, "%06zu", i);
        named.emplace_back(buf, params[i]);
    }
    return named;
}

} // namespace

// ── safetensors ──────────────────────────────────────────────────────────────

void save_safetensors(const string& path, const NamedTensors& tensors,
                      const map<string, string>& metadata) {
    if (!little_endian())
        throw runtime_error("save_safetensors: big-endian hosts are not supported");

    // contiguous copies in each tensor's own dtype: a view is written as its
    // values, not as the storage it happens to share
    vector<TensorPtr> data;
    for (auto& kv : tensors) {
        if (kv.first.empty()) throw runtime_error("save_safetensors: a tensor has an empty name");
        data.push_back(kv.second->detach());
    }
    {
        vector<string> names;
        for (auto& kv : tensors) names.push_back(kv.first);
        sort(names.begin(), names.end());
        if (adjacent_find(names.begin(), names.end()) != names.end())
            throw runtime_error("save_safetensors: duplicate tensor name");
    }

    string header = "{";
    bool first = true;
    if (!metadata.empty()) {
        header += "\"__metadata__\":{";
        bool f2 = true;
        for (auto& kv : metadata) {
            header += (f2 ? "" : ",") + string("\"") + json_escape(kv.first) + "\":\"" +
                      json_escape(kv.second) + "\"";
            f2 = false;
        }
        header += "}";
        first = false;
    }
    uint64_t offset = 0;
    for (size_t i = 0; i < tensors.size(); i++) {
        const auto& t = data[i];
        const uint64_t nbytes = (uint64_t)t->numel() * dtype_size(t->dtype());
        header += (first ? "" : ",") + string("\"") + json_escape(tensors[i].first) +
                  "\":{\"dtype\":\"" + st_name(t->dtype()) + "\",\"shape\":[";
        for (int d = 0; d < t->dim(); d++)
            header += (d ? "," : "") + to_string(t->shape()[d]);
        header += "],\"data_offsets\":[" + to_string(offset) + "," + to_string(offset + nbytes) + "]}";
        offset += nbytes;
        first = false;
    }
    header += "}";
    // the spec pads the header with spaces so the data starts 8-byte aligned
    while (header.size() % 8) header += ' ';

    ofstream file(path, ios::binary);
    if (!file)
        throw runtime_error("save_safetensors: cannot write '" + path + "' (does the directory exist?)");
    const uint64_t n = header.size();
    file.write(reinterpret_cast<const char*>(&n), 8);
    file.write(header.data(), (streamsize)header.size());
    for (auto& t : data)
        file.write(static_cast<const char*>(t->storage()->data()),
                   (streamsize)(t->numel() * dtype_size(t->dtype())));
    if (!file) throw runtime_error("save_safetensors: write to '" + path + "' failed");
}

NamedTensors load_safetensors(const string& path, map<string, string>* metadata) {
    if (!little_endian())
        throw runtime_error("load_safetensors: big-endian hosts are not supported");
    ifstream file(path, ios::binary | ios::ate);
    if (!file) throw runtime_error("load_safetensors: cannot open '" + path + "'");
    const uint64_t size = (uint64_t)file.tellg();
    file.seekg(0);

    uint64_t n = 0;
    file.read(reinterpret_cast<char*>(&n), 8);
    if (!file || n > MAX_HEADER || n > size - 8)
        throw runtime_error("load_safetensors: '" + path + "' is not a safetensors file");
    string header((size_t)n, '\0');
    file.read(&header[0], (streamsize)n);
    const uint64_t data_start = 8 + n, data_size = size - data_start;

    Json root = Parser(header, path).parse();
    if (root.kind != Json::Object) throw runtime_error("load_safetensors: '" + path + "': header is not an object");

    struct Entry { string name; DType dtype; Shape shape; uint64_t begin, end; };
    vector<Entry> entries;
    for (auto& kv : root.obj) {
        if (kv.first == "__metadata__") {
            if (kv.second.kind != Json::Object)
                throw runtime_error("load_safetensors: '" + path + "': __metadata__ is not an object");
            if (metadata)
                for (auto& m : kv.second.obj) {
                    if (m.second.kind != Json::String)
                        throw runtime_error("load_safetensors: '" + path + "': metadata values must be strings");
                    (*metadata)[m.first] = m.second.str;
                }
            continue;
        }
        const Json& t = kv.second;
        const Json* dt = t.get("dtype");
        const Json* sh = t.get("shape");
        const Json* off = t.get("data_offsets");
        if (t.kind != Json::Object || !dt || dt->kind != Json::String || !sh || sh->kind != Json::Array ||
            !off || off->kind != Json::Array || off->arr.size() != 2)
            throw runtime_error("load_safetensors: '" + path + "': tensor '" + kv.first + "' is malformed");
        Entry e{kv.first, st_dtype(dt->str, path), {}, off->arr[0].num, off->arr[1].num};
        uint64_t numel = 1;
        for (auto& d : sh->arr) {
            if (d.kind != Json::Number) throw runtime_error("load_safetensors: '" + path + "': bad shape");
            if (d.num && numel > numeric_limits<uint64_t>::max() / d.num)
                throw runtime_error("load_safetensors: '" + path + "': shape too large");
            numel *= d.num;
            e.shape.push_back((int64_t)d.num);
        }
        // every byte range must be exactly the tensor's size and inside the file
        if (e.end < e.begin || e.end > data_size || e.end - e.begin != numel * dtype_size(e.dtype))
            throw runtime_error("load_safetensors: '" + path + "': tensor '" + kv.first +
                                "' has a byte range that does not match its shape and dtype");
        entries.push_back(std::move(e));
    }
    sort(entries.begin(), entries.end(), [](const Entry& a, const Entry& b) { return a.begin < b.begin; });

    NamedTensors out;
    for (auto& e : entries) {
        auto t = make_shared<Tensor>(e.shape, e.dtype);
        file.seekg((streamoff)(data_start + e.begin));
        file.read(static_cast<char*>(t->storage()->data()), (streamsize)(e.end - e.begin));
        if (!file) throw runtime_error("load_safetensors: '" + path + "' is truncated");
        out.emplace_back(e.name, t);
    }
    return out;
}

// ── models ───────────────────────────────────────────────────────────────────

void save(const string& path, vector<TensorPtr> params) {
    save_safetensors(path, numbered(params), {{"format", "prometheus"}, {"kind", "model"}});
}

vector<TensorPtr> load(const string& path) {
    if (is_safetensors(path)) return in_order(load_safetensors(path));
    ifstream file(path, ios::binary);
    if (!file)
        throw runtime_error("load: cannot open '" + path + "' (paths are relative to the "
                            "current working directory)");
    int count = 0;
    file.read(reinterpret_cast<char*>(&count), sizeof(int));
    if (!file) throw runtime_error("load: '" + path + "' is not a valid model file");
    return read_legacy_tensors(file, path, count);
}

// ── checkpoints ──────────────────────────────────────────────────────────────

void save_checkpoint(const string& path, const vector<TensorPtr>& params, int epoch, float loss) {
    char buf[32];
    snprintf(buf, sizeof buf, "%.9g", loss);   // round-trips a float exactly
    save_safetensors(path, numbered(params),
                     {{"format", "prometheus"}, {"kind", "checkpoint"},
                      {"epoch", to_string(epoch)}, {"loss", buf}});
}

Checkpoint load_checkpoint(const string& path, vector<TensorPtr>& params) {
    Checkpoint ckpt{0, 0.0f};
    if (is_safetensors(path)) {
        map<string, string> meta;
        params = in_order(load_safetensors(path, &meta));
        if (!meta.count("epoch") || !meta.count("loss"))
            throw runtime_error("load_checkpoint: '" + path + "' has no epoch / loss metadata; "
                                "for plain weights use load()");
        ckpt.epoch = stoi(meta["epoch"]);
        ckpt.loss  = stof(meta["loss"]);
        return ckpt;
    }
    // the previous format: "CKPT", version 1, epoch, loss, then the tensors
    ifstream file(path, ios::binary);
    if (!file) throw runtime_error("load_checkpoint: cannot open " + path);
    char magic[4] = {0};
    file.read(magic, 4);
    if (memcmp(magic, "CKPT", 4) != 0)
        throw runtime_error("load_checkpoint: '" + path + "' is not a checkpoint");
    int version = 0, count = 0;
    file.read(reinterpret_cast<char*>(&version), sizeof(int));
    file.read(reinterpret_cast<char*>(&ckpt.epoch), sizeof(int));
    file.read(reinterpret_cast<char*>(&ckpt.loss), sizeof(float));
    file.read(reinterpret_cast<char*>(&count), sizeof(int));
    if (!file || version != 1)
        throw runtime_error("load_checkpoint: '" + path + "' has an unknown checkpoint version");
    params = read_legacy_tensors(file, path, count);
    return ckpt;
}
