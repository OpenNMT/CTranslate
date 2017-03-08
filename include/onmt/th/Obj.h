#pragma once

#include <string>
#include <map>
#include <set>
#include <list>
#include <vector>

#include "TH/THDiskFile.h"

namespace onmt
{
  namespace th
  {

    class Env;

    static const int dfLongSize = 8;

    enum class ObjType
    {
      NIL                   = 0,
      NUMBER                = 1,
      STRING                = 2,
      TABLE                 = 3,
      TORCH                 = 4,
      BOOLEAN               = 5,
      FUNCTION              = 6,
      RECUR_FUNCTION        = 8,
      LEGACY_RECUR_FUNCTION = 7
    };


    class Obj
    {
    public:
      Obj(ObjType type);
      virtual ~Obj() = default;
      virtual void read(THFile*, Env& Env) = 0;

      inline ObjType type() const
        {
          return _type;
        }
    private:
      const ObjType _type;
    };


    Obj* read_obj(THFile *tf, Env &Env);


    class Nil: public Obj
    {
    public:
      Nil();
      Nil(THFile*, Env& Env);
      void read(THFile*, Env& Env);
    };


    class Number: public Obj
    {
    public:
      Number();
      Number(THFile*, Env& Env);
      void read(THFile*, Env& Env);
      double get_value() const;
    private:
      double _value;
    };


    class Boolean: public Obj
    {
    public:
      Boolean();
      Boolean(THFile*, Env& Env);
      void read(THFile*, Env& Env);
      bool get_value() const;
    private:
      bool _value;
    };


    class String: public Obj
    {
    public:
      String();
      String(THFile*, Env& Env);
      void read(THFile*, Env& Env);
      const std::string& get_value() const;
    private:
      std::string _value;
    };


    class Table: public Obj {
    public:
      enum class TableType
      {
        None,
        Array,
        Object,
        Map
      };

      Table();
      Table(THFile*, Env& Env);
      void read(THFile*, Env& Env);
      Table& insert(Obj* key, Obj* value);

      const std::map<Obj*, Obj*>& get_map() const;
      const std::map<std::string, Obj*>& get_object() const;
      const std::vector<Obj*>& get_array() const;

    private:
      TableType _type;
      std::map<Obj*, Obj*> _map;
      std::map<std::string, Obj*> _object;
      std::vector<Obj*> _array;
    };


    class TorchObj: public Obj
    {
    public:
      TorchObj(const std::string& classname, int version);
      virtual ~TorchObj() = default;
      const std::string& get_classname() const;
    private:
      std::string _classname;
      int _version;
    };


    class Creator
    {
    public:
      Creator(const std::string& classname, int version);
      virtual TorchObj* create(const std::string& classname, int version) = 0;
    };


    template <typename T>
    class CreatorImpl: public Creator
    {
    public:
      CreatorImpl(const std::string& classname, int version)
        : Creator(classname, version)
        {
        }

      virtual TorchObj* create(const std::string& classname, int version)
        {
          return new T(classname, version);
        }
    };


    class Factory
    {
    public:
      static TorchObj* create(const std::string& classname, int version);
      static void register_it(const std::string& classname, int version, Creator* creator);
    private:
      static std::map<std::pair<std::string, int>, Creator*>& get_table();
    };


    // forward declaration
    template <typename T>
    class Tensor;


    template <typename T>
    class Storage: public TorchObj
    {
    public:
      Storage(const std::string &classname, int version);
      Storage(const std::string &classname, int version, THFile*, Env&);
      ~Storage();

      void read(THFile*, Env& Env);
      const T* get_data() const;
      long get_size() const;
      void release();

    private:
      friend class Tensor<T>;
      static const CreatorImpl< Storage<T> > creator;

      long _size;
      T *_data;
    };


    template <typename T>
    class Tensor: public TorchObj
    {
    public:
      Tensor(const std::string& classname, int version);
      Tensor(const std::string& classname, int version, THFile*, Env&);
      ~Tensor();

      void read(THFile*, Env&);
      Obj* get_storage() const;
      const long* get_size() const;
      int get_dimension() const;
      long get_storage_offset() const;
      void release_storage() const;
    private:
      static const CreatorImpl< Tensor<T> > creator;

      int _n_dimension;
      long* _size;
      long* _stride;
      long _storage_offset;
      Obj* _thstorage;
    };


    class Class: public TorchObj
    {
    public:
      Class(const std::string& classname, int version);
      Class(const std::string& classname, int version, THFile*, Env&);

      void read(THFile*, Env&);
      Obj* get_data() const;

    private:
      Obj* _data;
    };

  }
}

#include "Obj.hxx"
