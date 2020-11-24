// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: node_parameter.proto

#include "node_parameter.pb.h"

#include <algorithm>

#include <google/protobuf/stubs/common.h>
#include <google/protobuf/stubs/port.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/wire_format_lite_inl.h>
#include <google/protobuf/descriptor.h>
#include <google/protobuf/generated_message_reflection.h>
#include <google/protobuf/reflection_ops.h>
#include <google/protobuf/wire_format.h>
// This is a temporary google only hack
#ifdef GOOGLE_PROTOBUF_ENFORCE_UNIQUENESS
#include "third_party/protobuf/version.h"
#endif
// @@protoc_insertion_point(includes)

namespace intellgraph {
class NodeParameterDefaultTypeInternal {
 public:
  ::google::protobuf::internal::ExplicitlyConstructed<NodeParameter>
      _instance;
} _NodeParameter_default_instance_;
}  // namespace intellgraph
namespace protobuf_node_5fparameter_2eproto {
static void InitDefaultsNodeParameter() {
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  {
    void* ptr = &::intellgraph::_NodeParameter_default_instance_;
    new (ptr) ::intellgraph::NodeParameter();
    ::google::protobuf::internal::OnShutdownDestroyMessage(ptr);
  }
  ::intellgraph::NodeParameter::InitAsDefaultInstance();
}

::google::protobuf::internal::SCCInfo<0> scc_info_NodeParameter =
    {{ATOMIC_VAR_INIT(::google::protobuf::internal::SCCInfoBase::kUninitialized), 0, InitDefaultsNodeParameter}, {}};

void InitDefaults() {
  ::google::protobuf::internal::InitSCC(&scc_info_NodeParameter.base);
}

::google::protobuf::Metadata file_level_metadata[1];

const ::google::protobuf::uint32 TableStruct::offsets[] GOOGLE_PROTOBUF_ATTRIBUTE_SECTION_VARIABLE(protodesc_cold) = {
  ~0u,  // no _has_bits_
  GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(::intellgraph::NodeParameter, _internal_metadata_),
  ~0u,  // no _extensions_
  ~0u,  // no _oneof_case_
  ~0u,  // no _weak_field_map_
  GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(::intellgraph::NodeParameter, id_),
  GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(::intellgraph::NodeParameter, type_),
  GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(::intellgraph::NodeParameter, batch_size_),
  GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(::intellgraph::NodeParameter, dims_),
};
static const ::google::protobuf::internal::MigrationSchema schemas[] GOOGLE_PROTOBUF_ATTRIBUTE_SECTION_VARIABLE(protodesc_cold) = {
  { 0, -1, sizeof(::intellgraph::NodeParameter)},
};

static ::google::protobuf::Message const * const file_default_instances[] = {
  reinterpret_cast<const ::google::protobuf::Message*>(&::intellgraph::_NodeParameter_default_instance_),
};

void protobuf_AssignDescriptors() {
  AddDescriptors();
  AssignDescriptors(
      "node_parameter.proto", schemas, file_default_instances, TableStruct::offsets,
      file_level_metadata, NULL, NULL);
}

void protobuf_AssignDescriptorsOnce() {
  static ::google::protobuf::internal::once_flag once;
  ::google::protobuf::internal::call_once(once, protobuf_AssignDescriptors);
}

void protobuf_RegisterTypes(const ::std::string&) GOOGLE_PROTOBUF_ATTRIBUTE_COLD;
void protobuf_RegisterTypes(const ::std::string&) {
  protobuf_AssignDescriptorsOnce();
  ::google::protobuf::internal::RegisterAllTypes(file_level_metadata, 1);
}

void AddDescriptorsImpl() {
  InitDefaults();
  static const char descriptor[] GOOGLE_PROTOBUF_ATTRIBUTE_SECTION_VARIABLE(protodesc_cold) = {
      "\n\024node_parameter.proto\022\013intellgraph\"K\n\rN"
      "odeParameter\022\n\n\002id\030\001 \001(\005\022\014\n\004type\030\002 \001(\t\022\022"
      "\n\nbatch_size\030\003 \001(\005\022\014\n\004dims\030\004 \001(\005b\006proto3"
  };
  ::google::protobuf::DescriptorPool::InternalAddGeneratedFile(
      descriptor, 120);
  ::google::protobuf::MessageFactory::InternalRegisterGeneratedFile(
    "node_parameter.proto", &protobuf_RegisterTypes);
}

void AddDescriptors() {
  static ::google::protobuf::internal::once_flag once;
  ::google::protobuf::internal::call_once(once, AddDescriptorsImpl);
}
// Force AddDescriptors() to be called at dynamic initialization time.
struct StaticDescriptorInitializer {
  StaticDescriptorInitializer() {
    AddDescriptors();
  }
} static_descriptor_initializer;
}  // namespace protobuf_node_5fparameter_2eproto
namespace intellgraph {

// ===================================================================

void NodeParameter::InitAsDefaultInstance() {
}
#if !defined(_MSC_VER) || _MSC_VER >= 1900
const int NodeParameter::kIdFieldNumber;
const int NodeParameter::kTypeFieldNumber;
const int NodeParameter::kBatchSizeFieldNumber;
const int NodeParameter::kDimsFieldNumber;
#endif  // !defined(_MSC_VER) || _MSC_VER >= 1900

NodeParameter::NodeParameter()
  : ::google::protobuf::Message(), _internal_metadata_(NULL) {
  ::google::protobuf::internal::InitSCC(
      &protobuf_node_5fparameter_2eproto::scc_info_NodeParameter.base);
  SharedCtor();
  // @@protoc_insertion_point(constructor:intellgraph.NodeParameter)
}
NodeParameter::NodeParameter(const NodeParameter& from)
  : ::google::protobuf::Message(),
      _internal_metadata_(NULL) {
  _internal_metadata_.MergeFrom(from._internal_metadata_);
  type_.UnsafeSetDefault(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
  if (from.type().size() > 0) {
    type_.AssignWithDefault(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), from.type_);
  }
  ::memcpy(&id_, &from.id_,
    static_cast<size_t>(reinterpret_cast<char*>(&dims_) -
    reinterpret_cast<char*>(&id_)) + sizeof(dims_));
  // @@protoc_insertion_point(copy_constructor:intellgraph.NodeParameter)
}

void NodeParameter::SharedCtor() {
  type_.UnsafeSetDefault(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
  ::memset(&id_, 0, static_cast<size_t>(
      reinterpret_cast<char*>(&dims_) -
      reinterpret_cast<char*>(&id_)) + sizeof(dims_));
}

NodeParameter::~NodeParameter() {
  // @@protoc_insertion_point(destructor:intellgraph.NodeParameter)
  SharedDtor();
}

void NodeParameter::SharedDtor() {
  type_.DestroyNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
}

void NodeParameter::SetCachedSize(int size) const {
  _cached_size_.Set(size);
}
const ::google::protobuf::Descriptor* NodeParameter::descriptor() {
  ::protobuf_node_5fparameter_2eproto::protobuf_AssignDescriptorsOnce();
  return ::protobuf_node_5fparameter_2eproto::file_level_metadata[kIndexInFileMessages].descriptor;
}

const NodeParameter& NodeParameter::default_instance() {
  ::google::protobuf::internal::InitSCC(&protobuf_node_5fparameter_2eproto::scc_info_NodeParameter.base);
  return *internal_default_instance();
}


void NodeParameter::Clear() {
// @@protoc_insertion_point(message_clear_start:intellgraph.NodeParameter)
  ::google::protobuf::uint32 cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  type_.ClearToEmptyNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
  ::memset(&id_, 0, static_cast<size_t>(
      reinterpret_cast<char*>(&dims_) -
      reinterpret_cast<char*>(&id_)) + sizeof(dims_));
  _internal_metadata_.Clear();
}

bool NodeParameter::MergePartialFromCodedStream(
    ::google::protobuf::io::CodedInputStream* input) {
#define DO_(EXPRESSION) if (!GOOGLE_PREDICT_TRUE(EXPRESSION)) goto failure
  ::google::protobuf::uint32 tag;
  // @@protoc_insertion_point(parse_start:intellgraph.NodeParameter)
  for (;;) {
    ::std::pair<::google::protobuf::uint32, bool> p = input->ReadTagWithCutoffNoLastTag(127u);
    tag = p.first;
    if (!p.second) goto handle_unusual;
    switch (::google::protobuf::internal::WireFormatLite::GetTagFieldNumber(tag)) {
      // int32 id = 1;
      case 1: {
        if (static_cast< ::google::protobuf::uint8>(tag) ==
            static_cast< ::google::protobuf::uint8>(8u /* 8 & 0xFF */)) {

          DO_((::google::protobuf::internal::WireFormatLite::ReadPrimitive<
                   ::google::protobuf::int32, ::google::protobuf::internal::WireFormatLite::TYPE_INT32>(
                 input, &id_)));
        } else {
          goto handle_unusual;
        }
        break;
      }

      // string type = 2;
      case 2: {
        if (static_cast< ::google::protobuf::uint8>(tag) ==
            static_cast< ::google::protobuf::uint8>(18u /* 18 & 0xFF */)) {
          DO_(::google::protobuf::internal::WireFormatLite::ReadString(
                input, this->mutable_type()));
          DO_(::google::protobuf::internal::WireFormatLite::VerifyUtf8String(
            this->type().data(), static_cast<int>(this->type().length()),
            ::google::protobuf::internal::WireFormatLite::PARSE,
            "intellgraph.NodeParameter.type"));
        } else {
          goto handle_unusual;
        }
        break;
      }

      // int32 batch_size = 3;
      case 3: {
        if (static_cast< ::google::protobuf::uint8>(tag) ==
            static_cast< ::google::protobuf::uint8>(24u /* 24 & 0xFF */)) {

          DO_((::google::protobuf::internal::WireFormatLite::ReadPrimitive<
                   ::google::protobuf::int32, ::google::protobuf::internal::WireFormatLite::TYPE_INT32>(
                 input, &batch_size_)));
        } else {
          goto handle_unusual;
        }
        break;
      }

      // int32 dims = 4;
      case 4: {
        if (static_cast< ::google::protobuf::uint8>(tag) ==
            static_cast< ::google::protobuf::uint8>(32u /* 32 & 0xFF */)) {

          DO_((::google::protobuf::internal::WireFormatLite::ReadPrimitive<
                   ::google::protobuf::int32, ::google::protobuf::internal::WireFormatLite::TYPE_INT32>(
                 input, &dims_)));
        } else {
          goto handle_unusual;
        }
        break;
      }

      default: {
      handle_unusual:
        if (tag == 0) {
          goto success;
        }
        DO_(::google::protobuf::internal::WireFormat::SkipField(
              input, tag, _internal_metadata_.mutable_unknown_fields()));
        break;
      }
    }
  }
success:
  // @@protoc_insertion_point(parse_success:intellgraph.NodeParameter)
  return true;
failure:
  // @@protoc_insertion_point(parse_failure:intellgraph.NodeParameter)
  return false;
#undef DO_
}

void NodeParameter::SerializeWithCachedSizes(
    ::google::protobuf::io::CodedOutputStream* output) const {
  // @@protoc_insertion_point(serialize_start:intellgraph.NodeParameter)
  ::google::protobuf::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  // int32 id = 1;
  if (this->id() != 0) {
    ::google::protobuf::internal::WireFormatLite::WriteInt32(1, this->id(), output);
  }

  // string type = 2;
  if (this->type().size() > 0) {
    ::google::protobuf::internal::WireFormatLite::VerifyUtf8String(
      this->type().data(), static_cast<int>(this->type().length()),
      ::google::protobuf::internal::WireFormatLite::SERIALIZE,
      "intellgraph.NodeParameter.type");
    ::google::protobuf::internal::WireFormatLite::WriteStringMaybeAliased(
      2, this->type(), output);
  }

  // int32 batch_size = 3;
  if (this->batch_size() != 0) {
    ::google::protobuf::internal::WireFormatLite::WriteInt32(3, this->batch_size(), output);
  }

  // int32 dims = 4;
  if (this->dims() != 0) {
    ::google::protobuf::internal::WireFormatLite::WriteInt32(4, this->dims(), output);
  }

  if ((_internal_metadata_.have_unknown_fields() &&  ::google::protobuf::internal::GetProto3PreserveUnknownsDefault())) {
    ::google::protobuf::internal::WireFormat::SerializeUnknownFields(
        (::google::protobuf::internal::GetProto3PreserveUnknownsDefault()   ? _internal_metadata_.unknown_fields()   : _internal_metadata_.default_instance()), output);
  }
  // @@protoc_insertion_point(serialize_end:intellgraph.NodeParameter)
}

::google::protobuf::uint8* NodeParameter::InternalSerializeWithCachedSizesToArray(
    bool deterministic, ::google::protobuf::uint8* target) const {
  (void)deterministic; // Unused
  // @@protoc_insertion_point(serialize_to_array_start:intellgraph.NodeParameter)
  ::google::protobuf::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  // int32 id = 1;
  if (this->id() != 0) {
    target = ::google::protobuf::internal::WireFormatLite::WriteInt32ToArray(1, this->id(), target);
  }

  // string type = 2;
  if (this->type().size() > 0) {
    ::google::protobuf::internal::WireFormatLite::VerifyUtf8String(
      this->type().data(), static_cast<int>(this->type().length()),
      ::google::protobuf::internal::WireFormatLite::SERIALIZE,
      "intellgraph.NodeParameter.type");
    target =
      ::google::protobuf::internal::WireFormatLite::WriteStringToArray(
        2, this->type(), target);
  }

  // int32 batch_size = 3;
  if (this->batch_size() != 0) {
    target = ::google::protobuf::internal::WireFormatLite::WriteInt32ToArray(3, this->batch_size(), target);
  }

  // int32 dims = 4;
  if (this->dims() != 0) {
    target = ::google::protobuf::internal::WireFormatLite::WriteInt32ToArray(4, this->dims(), target);
  }

  if ((_internal_metadata_.have_unknown_fields() &&  ::google::protobuf::internal::GetProto3PreserveUnknownsDefault())) {
    target = ::google::protobuf::internal::WireFormat::SerializeUnknownFieldsToArray(
        (::google::protobuf::internal::GetProto3PreserveUnknownsDefault()   ? _internal_metadata_.unknown_fields()   : _internal_metadata_.default_instance()), target);
  }
  // @@protoc_insertion_point(serialize_to_array_end:intellgraph.NodeParameter)
  return target;
}

size_t NodeParameter::ByteSizeLong() const {
// @@protoc_insertion_point(message_byte_size_start:intellgraph.NodeParameter)
  size_t total_size = 0;

  if ((_internal_metadata_.have_unknown_fields() &&  ::google::protobuf::internal::GetProto3PreserveUnknownsDefault())) {
    total_size +=
      ::google::protobuf::internal::WireFormat::ComputeUnknownFieldsSize(
        (::google::protobuf::internal::GetProto3PreserveUnknownsDefault()   ? _internal_metadata_.unknown_fields()   : _internal_metadata_.default_instance()));
  }
  // string type = 2;
  if (this->type().size() > 0) {
    total_size += 1 +
      ::google::protobuf::internal::WireFormatLite::StringSize(
        this->type());
  }

  // int32 id = 1;
  if (this->id() != 0) {
    total_size += 1 +
      ::google::protobuf::internal::WireFormatLite::Int32Size(
        this->id());
  }

  // int32 batch_size = 3;
  if (this->batch_size() != 0) {
    total_size += 1 +
      ::google::protobuf::internal::WireFormatLite::Int32Size(
        this->batch_size());
  }

  // int32 dims = 4;
  if (this->dims() != 0) {
    total_size += 1 +
      ::google::protobuf::internal::WireFormatLite::Int32Size(
        this->dims());
  }

  int cached_size = ::google::protobuf::internal::ToCachedSize(total_size);
  SetCachedSize(cached_size);
  return total_size;
}

void NodeParameter::MergeFrom(const ::google::protobuf::Message& from) {
// @@protoc_insertion_point(generalized_merge_from_start:intellgraph.NodeParameter)
  GOOGLE_DCHECK_NE(&from, this);
  const NodeParameter* source =
      ::google::protobuf::internal::DynamicCastToGenerated<const NodeParameter>(
          &from);
  if (source == NULL) {
  // @@protoc_insertion_point(generalized_merge_from_cast_fail:intellgraph.NodeParameter)
    ::google::protobuf::internal::ReflectionOps::Merge(from, this);
  } else {
  // @@protoc_insertion_point(generalized_merge_from_cast_success:intellgraph.NodeParameter)
    MergeFrom(*source);
  }
}

void NodeParameter::MergeFrom(const NodeParameter& from) {
// @@protoc_insertion_point(class_specific_merge_from_start:intellgraph.NodeParameter)
  GOOGLE_DCHECK_NE(&from, this);
  _internal_metadata_.MergeFrom(from._internal_metadata_);
  ::google::protobuf::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  if (from.type().size() > 0) {

    type_.AssignWithDefault(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), from.type_);
  }
  if (from.id() != 0) {
    set_id(from.id());
  }
  if (from.batch_size() != 0) {
    set_batch_size(from.batch_size());
  }
  if (from.dims() != 0) {
    set_dims(from.dims());
  }
}

void NodeParameter::CopyFrom(const ::google::protobuf::Message& from) {
// @@protoc_insertion_point(generalized_copy_from_start:intellgraph.NodeParameter)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

void NodeParameter::CopyFrom(const NodeParameter& from) {
// @@protoc_insertion_point(class_specific_copy_from_start:intellgraph.NodeParameter)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

bool NodeParameter::IsInitialized() const {
  return true;
}

void NodeParameter::Swap(NodeParameter* other) {
  if (other == this) return;
  InternalSwap(other);
}
void NodeParameter::InternalSwap(NodeParameter* other) {
  using std::swap;
  type_.Swap(&other->type_, &::google::protobuf::internal::GetEmptyStringAlreadyInited(),
    GetArenaNoVirtual());
  swap(id_, other->id_);
  swap(batch_size_, other->batch_size_);
  swap(dims_, other->dims_);
  _internal_metadata_.Swap(&other->_internal_metadata_);
}

::google::protobuf::Metadata NodeParameter::GetMetadata() const {
  protobuf_node_5fparameter_2eproto::protobuf_AssignDescriptorsOnce();
  return ::protobuf_node_5fparameter_2eproto::file_level_metadata[kIndexInFileMessages];
}


// @@protoc_insertion_point(namespace_scope)
}  // namespace intellgraph
namespace google {
namespace protobuf {
template<> GOOGLE_PROTOBUF_ATTRIBUTE_NOINLINE ::intellgraph::NodeParameter* Arena::CreateMaybeMessage< ::intellgraph::NodeParameter >(Arena* arena) {
  return Arena::CreateInternal< ::intellgraph::NodeParameter >(arena);
}
}  // namespace protobuf
}  // namespace google

// @@protoc_insertion_point(global_scope)
