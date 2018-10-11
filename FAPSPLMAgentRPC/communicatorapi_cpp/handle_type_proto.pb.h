// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: handle_type_proto.proto

#ifndef PROTOBUF_INCLUDED_handle_5ftype_5fproto_2eproto
#define PROTOBUF_INCLUDED_handle_5ftype_5fproto_2eproto

#include <string>

#include <google/protobuf/stubs/common.h>

#if GOOGLE_PROTOBUF_VERSION < 3006001
#error This file was generated by a newer version of protoc which is
#error incompatible with your Protocol Buffer headers.  Please update
#error your headers.
#endif
#if 3006001 < GOOGLE_PROTOBUF_MIN_PROTOC_VERSION
#error This file was generated by an older version of protoc which is
#error incompatible with your Protocol Buffer headers.  Please
#error regenerate this file with a newer version of protoc.
#endif

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/arena.h>
#include <google/protobuf/arenastring.h>
#include <google/protobuf/generated_message_table_driven.h>
#include <google/protobuf/generated_message_util.h>
#include <google/protobuf/inlined_string_field.h>
#include <google/protobuf/metadata.h>
#include <google/protobuf/message.h>
#include <google/protobuf/repeated_field.h>  // IWYU pragma: export
#include <google/protobuf/extension_set.h>  // IWYU pragma: export
#include <google/protobuf/unknown_field_set.h>
// @@protoc_insertion_point(includes)
#define PROTOBUF_INTERNAL_EXPORT_protobuf_handle_5ftype_5fproto_2eproto 

namespace protobuf_handle_5ftype_5fproto_2eproto {
// Internal implementation detail -- do not use these members.
struct TableStruct {
  static const ::google::protobuf::internal::ParseTableField entries[];
  static const ::google::protobuf::internal::AuxillaryParseTableField aux[];
  static const ::google::protobuf::internal::ParseTable schema[1];
  static const ::google::protobuf::internal::FieldMetadata field_metadata[];
  static const ::google::protobuf::internal::SerializationTable serialization_table[];
  static const ::google::protobuf::uint32 offsets[];
};
void AddDescriptors();
}  // namespace protobuf_handle_5ftype_5fproto_2eproto
class HandleTypeProto;
class HandleTypeProtoDefaultTypeInternal;
extern HandleTypeProtoDefaultTypeInternal _HandleTypeProto_default_instance_;
namespace google {
namespace protobuf {
template<> ::HandleTypeProto* Arena::CreateMaybeMessage<::HandleTypeProto>(Arena*);
}  // namespace protobuf
}  // namespace google

// ===================================================================

class HandleTypeProto : public ::google::protobuf::Message /* @@protoc_insertion_point(class_definition:HandleTypeProto) */ {
 public:
  HandleTypeProto();
  virtual ~HandleTypeProto();

  HandleTypeProto(const HandleTypeProto& from);

  inline HandleTypeProto& operator=(const HandleTypeProto& from) {
    CopyFrom(from);
    return *this;
  }
  #if LANG_CXX11
  HandleTypeProto(HandleTypeProto&& from) noexcept
    : HandleTypeProto() {
    *this = ::std::move(from);
  }

  inline HandleTypeProto& operator=(HandleTypeProto&& from) noexcept {
    if (GetArenaNoVirtual() == from.GetArenaNoVirtual()) {
      if (this != &from) InternalSwap(&from);
    } else {
      CopyFrom(from);
    }
    return *this;
  }
  #endif
  static const ::google::protobuf::Descriptor* descriptor();
  static const HandleTypeProto& default_instance();

  static void InitAsDefaultInstance();  // FOR INTERNAL USE ONLY
  static inline const HandleTypeProto* internal_default_instance() {
    return reinterpret_cast<const HandleTypeProto*>(
               &_HandleTypeProto_default_instance_);
  }
  static constexpr int kIndexInFileMessages =
    0;

  void Swap(HandleTypeProto* other);
  friend void swap(HandleTypeProto& a, HandleTypeProto& b) {
    a.Swap(&b);
  }

  // implements Message ----------------------------------------------

  inline HandleTypeProto* New() const final {
    return CreateMaybeMessage<HandleTypeProto>(NULL);
  }

  HandleTypeProto* New(::google::protobuf::Arena* arena) const final {
    return CreateMaybeMessage<HandleTypeProto>(arena);
  }
  void CopyFrom(const ::google::protobuf::Message& from) final;
  void MergeFrom(const ::google::protobuf::Message& from) final;
  void CopyFrom(const HandleTypeProto& from);
  void MergeFrom(const HandleTypeProto& from);
  void Clear() final;
  bool IsInitialized() const final;

  size_t ByteSizeLong() const final;
  bool MergePartialFromCodedStream(
      ::google::protobuf::io::CodedInputStream* input) final;
  void SerializeWithCachedSizes(
      ::google::protobuf::io::CodedOutputStream* output) const final;
  ::google::protobuf::uint8* InternalSerializeWithCachedSizesToArray(
      bool deterministic, ::google::protobuf::uint8* target) const final;
  int GetCachedSize() const final { return _cached_size_.Get(); }

  private:
  void SharedCtor();
  void SharedDtor();
  void SetCachedSize(int size) const final;
  void InternalSwap(HandleTypeProto* other);
  private:
  inline ::google::protobuf::Arena* GetArenaNoVirtual() const {
    return NULL;
  }
  inline void* MaybeArenaPtr() const {
    return NULL;
  }
  public:

  ::google::protobuf::Metadata GetMetadata() const final;

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  // int32 handle = 1;
  void clear_handle();
  static const int kHandleFieldNumber = 1;
  ::google::protobuf::int32 handle() const;
  void set_handle(::google::protobuf::int32 value);

  // @@protoc_insertion_point(class_scope:HandleTypeProto)
 private:

  ::google::protobuf::internal::InternalMetadataWithArena _internal_metadata_;
  ::google::protobuf::int32 handle_;
  mutable ::google::protobuf::internal::CachedSize _cached_size_;
  friend struct ::protobuf_handle_5ftype_5fproto_2eproto::TableStruct;
};
// ===================================================================


// ===================================================================

#ifdef __GNUC__
  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif  // __GNUC__
// HandleTypeProto

// int32 handle = 1;
inline void HandleTypeProto::clear_handle() {
  handle_ = 0;
}
inline ::google::protobuf::int32 HandleTypeProto::handle() const {
  // @@protoc_insertion_point(field_get:HandleTypeProto.handle)
  return handle_;
}
inline void HandleTypeProto::set_handle(::google::protobuf::int32 value) {
  
  handle_ = value;
  // @@protoc_insertion_point(field_set:HandleTypeProto.handle)
}

#ifdef __GNUC__
  #pragma GCC diagnostic pop
#endif  // __GNUC__

// @@protoc_insertion_point(namespace_scope)


// @@protoc_insertion_point(global_scope)

#endif  // PROTOBUF_INCLUDED_handle_5ftype_5fproto_2eproto
