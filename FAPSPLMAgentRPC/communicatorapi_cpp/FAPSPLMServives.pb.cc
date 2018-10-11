// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: FAPSPLMServives.proto

#include "FAPSPLMServives.pb.h"

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

namespace communicator_objects {
}  // namespace communicator_objects
namespace protobuf_FAPSPLMServives_2eproto {
void InitDefaults() {
}

const ::google::protobuf::uint32 TableStruct::offsets[1] = {};
static const ::google::protobuf::internal::MigrationSchema* schemas = NULL;
static const ::google::protobuf::Message* const* file_default_instances = NULL;

void protobuf_AssignDescriptors() {
  AddDescriptors();
  AssignDescriptors(
      "FAPSPLMServives.proto", schemas, file_default_instances, TableStruct::offsets,
      NULL, NULL, NULL);
}

void protobuf_AssignDescriptorsOnce() {
  static ::google::protobuf::internal::once_flag once;
  ::google::protobuf::internal::call_once(once, protobuf_AssignDescriptors);
}

void protobuf_RegisterTypes(const ::std::string&) GOOGLE_PROTOBUF_ATTRIBUTE_COLD;
void protobuf_RegisterTypes(const ::std::string&) {
  protobuf_AssignDescriptorsOnce();
}

void AddDescriptorsImpl() {
  InitDefaults();
  static const char descriptor[] GOOGLE_PROTOBUF_ATTRIBUTE_SECTION_VARIABLE(protodesc_cold) = {
      "\n\025FAPSPLMServives.proto\022\024communicator_ob"
      "jects\032!academy_configuration_proto.proto"
      "\032\031academy_state_proto.proto\032\032academy_act"
      "ion_proto.proto\032\027handle_type_proto.proto"
      "2\264\002\n\017FAPSPLMServices\022=\n\024FAPSAGENT_Initia"
      "lize\022\023.AcademyConfigProto\032\020.HandleTypePr"
      "oto\0225\n\017FAPSAGENT_Clear\022\020.HandleTypeProto"
      "\032\020.HandleTypeProto\0225\n\017FAPSAGENT_Start\022\020."
      "HandleTypeProto\032\020.HandleTypeProto\0224\n\016FAP"
      "SAGENT_Stop\022\020.HandleTypeProto\032\020.HandleTy"
      "peProto\022>\n\023FAPSAGENT_getAction\022\022.Academy"
      "StateProto\032\023.AcademyActionProtob\006proto3"
  };
  ::google::protobuf::DescriptorPool::InternalAddGeneratedFile(
      descriptor, 479);
  ::google::protobuf::MessageFactory::InternalRegisterGeneratedFile(
    "FAPSPLMServives.proto", &protobuf_RegisterTypes);
  ::protobuf_academy_5fconfiguration_5fproto_2eproto::AddDescriptors();
  ::protobuf_academy_5fstate_5fproto_2eproto::AddDescriptors();
  ::protobuf_academy_5faction_5fproto_2eproto::AddDescriptors();
  ::protobuf_handle_5ftype_5fproto_2eproto::AddDescriptors();
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
}  // namespace protobuf_FAPSPLMServives_2eproto
namespace communicator_objects {

// @@protoc_insertion_point(namespace_scope)
}  // namespace communicator_objects
namespace google {
namespace protobuf {
}  // namespace protobuf
}  // namespace google

// @@protoc_insertion_point(global_scope)
