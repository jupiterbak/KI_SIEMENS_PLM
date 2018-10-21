// <auto-generated>
//     Generated by the protocol buffer compiler.  DO NOT EDIT!
//     source: academy_action_proto.proto
// </auto-generated>
#pragma warning disable 1591, 0612, 3021
#region Designer generated code

using pb = global::Google.Protobuf;
using pbc = global::Google.Protobuf.Collections;
using pbr = global::Google.Protobuf.Reflection;
using scg = global::System.Collections.Generic;
/// <summary>Holder for reflection information generated from academy_action_proto.proto</summary>
public static partial class AcademyActionProtoReflection {

  #region Descriptor
  /// <summary>File descriptor for academy_action_proto.proto</summary>
  public static pbr::FileDescriptor Descriptor {
    get { return descriptor; }
  }
  private static pbr::FileDescriptor descriptor;

  static AcademyActionProtoReflection() {
    byte[] descriptorData = global::System.Convert.FromBase64String(
        string.Concat(
          "ChphY2FkZW15X2FjdGlvbl9wcm90by5wcm90bxoYYnJhaW5fYWN0aW9uX3By",
          "b3RvLnByb3RvGhdoYW5kbGVfdHlwZV9wcm90by5wcm90byKDAQoSQWNhZGVt",
          "eUFjdGlvblByb3RvEhMKC0FjYWRlbXlOYW1lGAEgASgJEhIKCmJyYWluQ291",
          "bnQYAiABKAUSIgoHYWN0aW9ucxgDIAMoCzIRLkJyYWluQWN0aW9uUHJvdG8S",
          "IAoGaGFuZGxlGAQgASgLMhAuSGFuZGxlVHlwZVByb3RvYgZwcm90bzM="));
    descriptor = pbr::FileDescriptor.FromGeneratedCode(descriptorData,
        new pbr::FileDescriptor[] { global::BrainActionProtoReflection.Descriptor, global::HandleTypeProtoReflection.Descriptor, },
        new pbr::GeneratedClrTypeInfo(null, new pbr::GeneratedClrTypeInfo[] {
          new pbr::GeneratedClrTypeInfo(typeof(global::AcademyActionProto), global::AcademyActionProto.Parser, new[]{ "AcademyName", "BrainCount", "Actions", "Handle" }, null, null, null)
        }));
  }
  #endregion

}
#region Messages
public sealed partial class AcademyActionProto : pb::IMessage<AcademyActionProto> {
  private static readonly pb::MessageParser<AcademyActionProto> _parser = new pb::MessageParser<AcademyActionProto>(() => new AcademyActionProto());
  private pb::UnknownFieldSet _unknownFields;
  [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
  public static pb::MessageParser<AcademyActionProto> Parser { get { return _parser; } }

  [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
  public static pbr::MessageDescriptor Descriptor {
    get { return global::AcademyActionProtoReflection.Descriptor.MessageTypes[0]; }
  }

  [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
  pbr::MessageDescriptor pb::IMessage.Descriptor {
    get { return Descriptor; }
  }

  [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
  public AcademyActionProto() {
    OnConstruction();
  }

  partial void OnConstruction();

  [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
  public AcademyActionProto(AcademyActionProto other) : this() {
    academyName_ = other.academyName_;
    brainCount_ = other.brainCount_;
    actions_ = other.actions_.Clone();
    handle_ = other.handle_ != null ? other.handle_.Clone() : null;
    _unknownFields = pb::UnknownFieldSet.Clone(other._unknownFields);
  }

  [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
  public AcademyActionProto Clone() {
    return new AcademyActionProto(this);
  }

  /// <summary>Field number for the "AcademyName" field.</summary>
  public const int AcademyNameFieldNumber = 1;
  private string academyName_ = "";
  [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
  public string AcademyName {
    get { return academyName_; }
    set {
      academyName_ = pb::ProtoPreconditions.CheckNotNull(value, "value");
    }
  }

  /// <summary>Field number for the "brainCount" field.</summary>
  public const int BrainCountFieldNumber = 2;
  private int brainCount_;
  [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
  public int BrainCount {
    get { return brainCount_; }
    set {
      brainCount_ = value;
    }
  }

  /// <summary>Field number for the "actions" field.</summary>
  public const int ActionsFieldNumber = 3;
  private static readonly pb::FieldCodec<global::BrainActionProto> _repeated_actions_codec
      = pb::FieldCodec.ForMessage(26, global::BrainActionProto.Parser);
  private readonly pbc::RepeatedField<global::BrainActionProto> actions_ = new pbc::RepeatedField<global::BrainActionProto>();
  [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
  public pbc::RepeatedField<global::BrainActionProto> Actions {
    get { return actions_; }
  }

  /// <summary>Field number for the "handle" field.</summary>
  public const int HandleFieldNumber = 4;
  private global::HandleTypeProto handle_;
  [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
  public global::HandleTypeProto Handle {
    get { return handle_; }
    set {
      handle_ = value;
    }
  }

  [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
  public override bool Equals(object other) {
    return Equals(other as AcademyActionProto);
  }

  [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
  public bool Equals(AcademyActionProto other) {
    if (ReferenceEquals(other, null)) {
      return false;
    }
    if (ReferenceEquals(other, this)) {
      return true;
    }
    if (AcademyName != other.AcademyName) return false;
    if (BrainCount != other.BrainCount) return false;
    if(!actions_.Equals(other.actions_)) return false;
    if (!object.Equals(Handle, other.Handle)) return false;
    return Equals(_unknownFields, other._unknownFields);
  }

  [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
  public override int GetHashCode() {
    int hash = 1;
    if (AcademyName.Length != 0) hash ^= AcademyName.GetHashCode();
    if (BrainCount != 0) hash ^= BrainCount.GetHashCode();
    hash ^= actions_.GetHashCode();
    if (handle_ != null) hash ^= Handle.GetHashCode();
    if (_unknownFields != null) {
      hash ^= _unknownFields.GetHashCode();
    }
    return hash;
  }

  [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
  public override string ToString() {
    return pb::JsonFormatter.ToDiagnosticString(this);
  }

  [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
  public void WriteTo(pb::CodedOutputStream output) {
    if (AcademyName.Length != 0) {
      output.WriteRawTag(10);
      output.WriteString(AcademyName);
    }
    if (BrainCount != 0) {
      output.WriteRawTag(16);
      output.WriteInt32(BrainCount);
    }
    actions_.WriteTo(output, _repeated_actions_codec);
    if (handle_ != null) {
      output.WriteRawTag(34);
      output.WriteMessage(Handle);
    }
    if (_unknownFields != null) {
      _unknownFields.WriteTo(output);
    }
  }

  [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
  public int CalculateSize() {
    int size = 0;
    if (AcademyName.Length != 0) {
      size += 1 + pb::CodedOutputStream.ComputeStringSize(AcademyName);
    }
    if (BrainCount != 0) {
      size += 1 + pb::CodedOutputStream.ComputeInt32Size(BrainCount);
    }
    size += actions_.CalculateSize(_repeated_actions_codec);
    if (handle_ != null) {
      size += 1 + pb::CodedOutputStream.ComputeMessageSize(Handle);
    }
    if (_unknownFields != null) {
      size += _unknownFields.CalculateSize();
    }
    return size;
  }

  [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
  public void MergeFrom(AcademyActionProto other) {
    if (other == null) {
      return;
    }
    if (other.AcademyName.Length != 0) {
      AcademyName = other.AcademyName;
    }
    if (other.BrainCount != 0) {
      BrainCount = other.BrainCount;
    }
    actions_.Add(other.actions_);
    if (other.handle_ != null) {
      if (handle_ == null) {
        handle_ = new global::HandleTypeProto();
      }
      Handle.MergeFrom(other.Handle);
    }
    _unknownFields = pb::UnknownFieldSet.MergeFrom(_unknownFields, other._unknownFields);
  }

  [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
  public void MergeFrom(pb::CodedInputStream input) {
    uint tag;
    while ((tag = input.ReadTag()) != 0) {
      switch(tag) {
        default:
          _unknownFields = pb::UnknownFieldSet.MergeFieldFrom(_unknownFields, input);
          break;
        case 10: {
          AcademyName = input.ReadString();
          break;
        }
        case 16: {
          BrainCount = input.ReadInt32();
          break;
        }
        case 26: {
          actions_.AddEntriesFrom(input, _repeated_actions_codec);
          break;
        }
        case 34: {
          if (handle_ == null) {
            handle_ = new global::HandleTypeProto();
          }
          input.ReadMessage(handle_);
          break;
        }
      }
    }
  }

}

#endregion


#endregion Designer generated code