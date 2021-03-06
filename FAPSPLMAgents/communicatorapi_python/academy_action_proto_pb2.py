# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: academy_action_proto.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


import FAPSPLMAgents.communicatorapi_python.brain_action_proto_pb2 as brain__action__proto__pb2
import FAPSPLMAgents.communicatorapi_python.handle_type_proto_pb2 as handle__type__proto__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='academy_action_proto.proto',
  package='',
  syntax='proto3',
  serialized_options=None,
  serialized_pb=_b('\n\x1a\x61\x63\x61\x64\x65my_action_proto.proto\x1a\x18\x62rain_action_proto.proto\x1a\x17handle_type_proto.proto\"\x83\x01\n\x12\x41\x63\x61\x64\x65myActionProto\x12\x13\n\x0b\x41\x63\x61\x64\x65myName\x18\x01 \x01(\t\x12\x12\n\nbrainCount\x18\x02 \x01(\x05\x12\"\n\x07\x61\x63tions\x18\x03 \x03(\x0b\x32\x11.BrainActionProto\x12 \n\x06handle\x18\x04 \x01(\x0b\x32\x10.HandleTypeProtob\x06proto3')
  ,
  dependencies=[brain__action__proto__pb2.DESCRIPTOR,handle__type__proto__pb2.DESCRIPTOR,])




_ACADEMYACTIONPROTO = _descriptor.Descriptor(
  name='AcademyActionProto',
  full_name='AcademyActionProto',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='AcademyName', full_name='AcademyActionProto.AcademyName', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='brainCount', full_name='AcademyActionProto.brainCount', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='actions', full_name='AcademyActionProto.actions', index=2,
      number=3, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='handle', full_name='AcademyActionProto.handle', index=3,
      number=4, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=82,
  serialized_end=213,
)

_ACADEMYACTIONPROTO.fields_by_name['actions'].message_type = brain__action__proto__pb2._BRAINACTIONPROTO
_ACADEMYACTIONPROTO.fields_by_name['handle'].message_type = handle__type__proto__pb2._HANDLETYPEPROTO
DESCRIPTOR.message_types_by_name['AcademyActionProto'] = _ACADEMYACTIONPROTO
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

AcademyActionProto = _reflection.GeneratedProtocolMessageType('AcademyActionProto', (_message.Message,), dict(
  DESCRIPTOR = _ACADEMYACTIONPROTO,
  __module__ = 'academy_action_proto_pb2'
  # @@protoc_insertion_point(class_scope:AcademyActionProto)
  ))
_sym_db.RegisterMessage(AcademyActionProto)


# @@protoc_insertion_point(module_scope)
