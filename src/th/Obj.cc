#include "onmt/th/Obj.h"

#include <iostream>

#include "onmt/th/Env.h"

#define REGISTERType(Name, version, Type)                               \
  template<>                                                            \
  const CreatorImpl<Tensor<Type> >                                      \
  Tensor<Type>::creator("torch." #Name "Tensor", version);              \
                                                                        \
  template<>                                                            \
  const CreatorImpl<Storage<Type> >                                     \
  Storage<Type>::creator("torch." #Name "Storage", version);            \
                                                                        \
  template<>                                                            \
  void Storage<Type>::read(THFile *tf, Env&)                            \
  {                                                                     \
    if (!tf)                                                            \
      return;                                                           \
    _size = THFile_readLongScalar(tf);                                  \
    _data = reinterpret_cast<Type*>(THAlloc(sizeof (Type) * _size));    \
    THFile_read ## Name ## Raw(tf, _data, _size);                       \
  }                                                                     \
                                                                        \
  Tensor<Type> ITensor ## Name(#Name, version);                         \
  Storage<Type> IStorage ## Name(#Name, version);                       \
  void read ## Tensor ## Name(Tensor<Type> &A, THFile* tf, Env& env)    \
  {                                                                     \
    A.read(tf, env);                                                    \
  }                                                                     \
  void read ## Storage ## Name(Tensor<Type> &A, THFile* tf, Env& env)   \
  {                                                                     \
    A.read(tf, env);                                                    \
  }

namespace onmt
{
  namespace th
  {

    REGISTERType(Double, 1, double);
    REGISTERType(Float, 1, float);
    REGISTERType(Byte, 1, unsigned char);
    REGISTERType(Char, 1, char);
    REGISTERType(Long, 1, long);
    REGISTERType(Short, 1, short);
    REGISTERType(Int, 1, int);

    TorchObj* Factory::create(const std::string& classname, int version)
    {
      auto i = get_table().find(std::make_pair(classname, version));

      if (i != get_table().end())
        return i->second->create(classname, version);
      else
        return nullptr;
    }

    void Factory::register_it(const std::string& classname, int version, Creator* thcreator)
    {
      get_table()[std::make_pair(classname, version)] = thcreator;
    }

    std::map<std::pair<std::string, int>, Creator*>& Factory::get_table()
    {
      static std::map<std::pair<std::string, int>, Creator*> table;
      return table;
    }

    Creator::Creator(const std::string& classname, int version)
    {
      Factory::register_it(classname, version, this);
    }


    Obj::Obj(ObjType type)
      : _type(type)
    {
    }


    Nil::Nil()
      : Obj(ObjType::NIL)
    {
    }

    Nil::Nil(THFile* tf, Env& env)
      : Obj(ObjType::NIL)
    {
      read(tf, env);
    }

    void Nil::read(THFile*, Env&)
    {
    }


    Number::Number()
      : Obj(ObjType::NUMBER)
      , _value(0)
    {
    }

    Number::Number(THFile* tf, Env& env)
      : Obj(ObjType::NUMBER)
    {
      read(tf, env);
    }

    void Number::read(THFile*tf, Env&)
    {
      _value=THFile_readDoubleScalar(tf);
    }

    double Number::get_value() const
    {
      return _value;
    }


    Boolean::Boolean()
      : Obj(ObjType::BOOLEAN)
      , _value(0)
    {
    }

    Boolean::Boolean(THFile* tf, Env& env)
      : Obj(ObjType::BOOLEAN)
    {
      read(tf, env);
    }

    void Boolean::read(THFile* tf, Env&)
    {
      _value=THFile_readIntScalar(tf) == 1;
    }

    bool Boolean::get_value() const
    {
      return _value;
    }


    String::String()
      : Obj(ObjType::STRING)
    {
    }

    String::String(THFile* tf, Env& env)
      : Obj(ObjType::STRING)
    {
      read(tf, env);
    }

    void String::read(THFile*tf, Env&)
    {
      int size = THFile_readIntScalar(tf);
      unsigned char *buffer = reinterpret_cast<unsigned char*>(THAlloc(size));
      THFile_readByteRaw(tf, buffer, size);
      _value = std::string(reinterpret_cast<char*>(buffer), size);
      THFree(buffer);
    }

    const std::string& String::get_value() const
    {
      return _value;
    }


    Table::Table()
      : Obj(ObjType::TABLE)
      , _type(TableType::None)
    {
    }

    Table::Table(THFile* tf, Env& env)
      : Obj(ObjType::TABLE)
      , _type(TableType::None)
    {
      read(tf, env);
    }

    void Table::read(THFile* tf, Env& env)
    {
      int size = THFile_readIntScalar(tf);

      for (int i = 0; i < size; i++)
      {
        Obj* key = read_obj(tf, env);
        Obj* value = read_obj(tf, env);
        insert(key, value);
      }
    }

    Table& Table::insert(Obj* key, Obj* value)
    {
      if (_type == TableType::None)
      {
        switch (key->type())
        {
        case ObjType::NUMBER:
          _type = TableType::Array;
          break;
        case ObjType::STRING:
          _type = TableType::Object;
          break;
        default:
          _type = TableType::Map;
          break;
        }
      }

      switch (_type)
      {
      case TableType::Array:
        _array.push_back(value);
        break;
      case TableType::Object:
      {
        String* key_str = dynamic_cast<String*>(key);
        if (key_str)
          _object[key_str->get_value()] = value;
        break;
      }
      default:
        _map[key] = value;
        break;
      }

      return *this;
    }

    const std::map<Obj*, Obj*>& Table::get_map() const
    {
      return _map;
    }

    const std::map<std::string, Obj*>& Table::get_object() const
    {
      return _object;
    }

    const std::vector<Obj*>& Table::get_array() const
    {
      return _array;
    }


    TorchObj::TorchObj(const std::string& classname, int version)
      : Obj(ObjType::TORCH)
      , _classname(classname)
      , _version(version)
    {
    }

    const std::string& TorchObj::get_classname() const
    {
      return _classname;
    }


    Class::Class(const std::string& classname, int version)
      : TorchObj(classname, version)
      , _data(0)
    {
    }

    Class::Class(const std::string& classname, int version, THFile* tf, Env& env)
      : TorchObj(classname, version)
    {
      read(tf, env);
    }

    void Class::read(THFile* tf, Env& env)
    {
      _data = read_obj(tf, env);
    }

    Obj* Class::get_data() const
    {
      return _data;
    }

    Obj* read_obj(THFile* tf, Env& env)
    {
      int tobj = THFile_readIntScalar(tf);

      switch (static_cast<ObjType>(tobj))
      {
      case ObjType::RECUR_FUNCTION:
      case ObjType::LEGACY_RECUR_FUNCTION:
      {
        // Read but ignore functions.
        int index = THFile_readIntScalar(tf);
        Obj* obj = env.get_object(index);

        if (obj)
          return obj;

        int size = THFile_readIntScalar(tf);
        unsigned char buffer[size];
        THFile_readByteRaw(tf, buffer, size);
        read_obj(tf, env);

        Nil* nil = new Nil();
        env.set_object(nil, index);
        return nil;
      }
      case ObjType::NIL:
      {
        Nil* nil = new Nil(tf, env);
        env.set_object(nil);
        return nil;
      }
      case ObjType::NUMBER:
      {
        Number* number = new Number(tf, env);
        env.set_object(number);
        return number;
      }
      case ObjType::BOOLEAN:
      {
        Boolean* boolean = new Boolean(tf, env);
        env.set_object(boolean);
        return boolean;
      }
      case ObjType::STRING:
      {
        String* string = new String(tf, env);
        env.set_object(string);
        return string;
      }
      case ObjType::TABLE:
      case ObjType::TORCH:
      {
        int index = THFile_readIntScalar(tf);
        Obj* obj = env.get_object(index);

        if (obj)
          return obj;

        if (static_cast<ObjType>(tobj) == ObjType::TABLE)
        {
          Table* table = new Table();
          env.set_object(table, index);
          table->read(tf, env);
          return table;
        }

        /* ObjType::TORCH */
        int size = THFile_readIntScalar(tf);
        unsigned char* buffer = reinterpret_cast<unsigned char*>(THAlloc(size));
        THFile_readByteRaw(tf, buffer, size);
        std::string classname;
        int version = 0;

        if (size == 3 && buffer[0] == 'V' && buffer[1] == ' ')
        {
          version = buffer[2] - '0';
          size = THFile_readIntScalar(tf);
          THFree(buffer);
          buffer = reinterpret_cast<unsigned char*>(THAlloc(size));
          THFile_readByteRaw(tf, buffer, size);
          classname = std::string(reinterpret_cast<char*>(buffer), size);
        }
        else
        {
          classname = version;
        }

        THFree(buffer);

        /* torch.XXXXTensor */
        if (classname.length() > 6 && classname.substr(0,6) == "torch.")
        {
          Obj* thtensor = Factory::create(classname, version);
          THAssert(thtensor);
          env.set_object(thtensor, index);
          thtensor->read(tf, env);
          return thtensor;
        }
        else
        {
          Class* thclass = new Class(classname, version);
          env.set_object(thclass, index);
          thclass->read(tf, env);
          return thclass;
        }

        std::cerr << "undefined classname=" << classname << std::endl;
        THAssert(0);
      }
      default:
        break;
      }

      std::cerr << "undefined object=" << tobj << std::endl;

      THAssert(0);
      return nullptr;
    }

  }
}
