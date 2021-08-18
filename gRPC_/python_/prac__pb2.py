# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: prac_.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='prac_.proto',
  package='practice',
  syntax='proto3',
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_pb=b'\n\x0bprac_.proto\x12\x08practice\"\x19\n\tTD_create\x12\x0c\n\x04Todo\x18\x01 \x01(\t\"\x1d\n\ncomplete_C\x12\x0f\n\x07message\x18\x01 \x01(\t\"\x19\n\tTD_remove\x12\x0c\n\x04Todo\x18\x01 \x01(\t\"\x1d\n\ncomplete_D\x12\x0f\n\x07message\x18\x01 \x01(\t\"\x19\n\tTD_update\x12\x0c\n\x04Todo\x18\x01 \x01(\t\"\x1d\n\ncomplete_U\x12\x0f\n\x07message\x18\x01 \x01(\t\"\x17\n\x07TD_read\x12\x0c\n\x04Todo\x18\x01 \x01(\t\"\x1a\n\ncomplete_R\x12\x0c\n\x04list\x18\x01 \x01(\t2\xe6\x01\n\x08TODO_APP\x12\x36\n\tCreate_TD\x12\x13.practice.TD_create\x1a\x14.practice.complete_C\x12\x36\n\tRemove_TD\x12\x13.practice.TD_remove\x1a\x14.practice.complete_D\x12\x36\n\tUpdate_TD\x12\x13.practice.TD_update\x1a\x14.practice.complete_U\x12\x32\n\x07Read_TD\x12\x11.practice.TD_read\x1a\x14.practice.complete_Rb\x06proto3'
)




_TD_CREATE = _descriptor.Descriptor(
  name='TD_create',
  full_name='practice.TD_create',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='Todo', full_name='practice.TD_create.Todo', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
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
  serialized_start=25,
  serialized_end=50,
)


_COMPLETE_C = _descriptor.Descriptor(
  name='complete_C',
  full_name='practice.complete_C',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='message', full_name='practice.complete_C.message', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
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
  serialized_start=52,
  serialized_end=81,
)


_TD_REMOVE = _descriptor.Descriptor(
  name='TD_remove',
  full_name='practice.TD_remove',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='Todo', full_name='practice.TD_remove.Todo', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
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
  serialized_start=83,
  serialized_end=108,
)


_COMPLETE_D = _descriptor.Descriptor(
  name='complete_D',
  full_name='practice.complete_D',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='message', full_name='practice.complete_D.message', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
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
  serialized_start=110,
  serialized_end=139,
)


_TD_UPDATE = _descriptor.Descriptor(
  name='TD_update',
  full_name='practice.TD_update',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='Todo', full_name='practice.TD_update.Todo', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
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
  serialized_start=141,
  serialized_end=166,
)


_COMPLETE_U = _descriptor.Descriptor(
  name='complete_U',
  full_name='practice.complete_U',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='message', full_name='practice.complete_U.message', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
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
  serialized_start=168,
  serialized_end=197,
)


_TD_READ = _descriptor.Descriptor(
  name='TD_read',
  full_name='practice.TD_read',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='Todo', full_name='practice.TD_read.Todo', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
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
  serialized_start=199,
  serialized_end=222,
)


_COMPLETE_R = _descriptor.Descriptor(
  name='complete_R',
  full_name='practice.complete_R',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='list', full_name='practice.complete_R.list', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
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
  serialized_start=224,
  serialized_end=250,
)

DESCRIPTOR.message_types_by_name['TD_create'] = _TD_CREATE
DESCRIPTOR.message_types_by_name['complete_C'] = _COMPLETE_C
DESCRIPTOR.message_types_by_name['TD_remove'] = _TD_REMOVE
DESCRIPTOR.message_types_by_name['complete_D'] = _COMPLETE_D
DESCRIPTOR.message_types_by_name['TD_update'] = _TD_UPDATE
DESCRIPTOR.message_types_by_name['complete_U'] = _COMPLETE_U
DESCRIPTOR.message_types_by_name['TD_read'] = _TD_READ
DESCRIPTOR.message_types_by_name['complete_R'] = _COMPLETE_R
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

TD_create = _reflection.GeneratedProtocolMessageType('TD_create', (_message.Message,), {
  'DESCRIPTOR' : _TD_CREATE,
  '__module__' : 'prac__pb2'
  # @@protoc_insertion_point(class_scope:practice.TD_create)
  })
_sym_db.RegisterMessage(TD_create)

complete_C = _reflection.GeneratedProtocolMessageType('complete_C', (_message.Message,), {
  'DESCRIPTOR' : _COMPLETE_C,
  '__module__' : 'prac__pb2'
  # @@protoc_insertion_point(class_scope:practice.complete_C)
  })
_sym_db.RegisterMessage(complete_C)

TD_remove = _reflection.GeneratedProtocolMessageType('TD_remove', (_message.Message,), {
  'DESCRIPTOR' : _TD_REMOVE,
  '__module__' : 'prac__pb2'
  # @@protoc_insertion_point(class_scope:practice.TD_remove)
  })
_sym_db.RegisterMessage(TD_remove)

complete_D = _reflection.GeneratedProtocolMessageType('complete_D', (_message.Message,), {
  'DESCRIPTOR' : _COMPLETE_D,
  '__module__' : 'prac__pb2'
  # @@protoc_insertion_point(class_scope:practice.complete_D)
  })
_sym_db.RegisterMessage(complete_D)

TD_update = _reflection.GeneratedProtocolMessageType('TD_update', (_message.Message,), {
  'DESCRIPTOR' : _TD_UPDATE,
  '__module__' : 'prac__pb2'
  # @@protoc_insertion_point(class_scope:practice.TD_update)
  })
_sym_db.RegisterMessage(TD_update)

complete_U = _reflection.GeneratedProtocolMessageType('complete_U', (_message.Message,), {
  'DESCRIPTOR' : _COMPLETE_U,
  '__module__' : 'prac__pb2'
  # @@protoc_insertion_point(class_scope:practice.complete_U)
  })
_sym_db.RegisterMessage(complete_U)

TD_read = _reflection.GeneratedProtocolMessageType('TD_read', (_message.Message,), {
  'DESCRIPTOR' : _TD_READ,
  '__module__' : 'prac__pb2'
  # @@protoc_insertion_point(class_scope:practice.TD_read)
  })
_sym_db.RegisterMessage(TD_read)

complete_R = _reflection.GeneratedProtocolMessageType('complete_R', (_message.Message,), {
  'DESCRIPTOR' : _COMPLETE_R,
  '__module__' : 'prac__pb2'
  # @@protoc_insertion_point(class_scope:practice.complete_R)
  })
_sym_db.RegisterMessage(complete_R)



_TODO_APP = _descriptor.ServiceDescriptor(
  name='TODO_APP',
  full_name='practice.TODO_APP',
  file=DESCRIPTOR,
  index=0,
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_start=253,
  serialized_end=483,
  methods=[
  _descriptor.MethodDescriptor(
    name='Create_TD',
    full_name='practice.TODO_APP.Create_TD',
    index=0,
    containing_service=None,
    input_type=_TD_CREATE,
    output_type=_COMPLETE_C,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='Remove_TD',
    full_name='practice.TODO_APP.Remove_TD',
    index=1,
    containing_service=None,
    input_type=_TD_REMOVE,
    output_type=_COMPLETE_D,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='Update_TD',
    full_name='practice.TODO_APP.Update_TD',
    index=2,
    containing_service=None,
    input_type=_TD_UPDATE,
    output_type=_COMPLETE_U,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='Read_TD',
    full_name='practice.TODO_APP.Read_TD',
    index=3,
    containing_service=None,
    input_type=_TD_READ,
    output_type=_COMPLETE_R,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
])
_sym_db.RegisterServiceDescriptor(_TODO_APP)

DESCRIPTOR.services_by_name['TODO_APP'] = _TODO_APP

# @@protoc_insertion_point(module_scope)