require "torch"

CSVLogger = {}
CSVLogger.__index = CSVLogger

function CSVLogger.create(file_path, fields)
	local self = {}
	setmetatable(self, CSVLogger)
	self.file = io.open(file_path, "a")

	self.fields  = {}
	self.indices = {}
	self.cur_row = {}
	if fields ~= nil then self:add_fields(fields) end
	return self
end

function CSVLogger:add_fields(fields)
	for k = 1, #fields do
		self.fields[#self.fields + 1] = fields[k]
		self.indices[fields[k]] = #self.fields
		self.cur_row[#self.cur_row + 1] = "\"\""
	end
end

function CSVLogger:write_header()
	self.file:write(table.concat(self.fields, ", "), "\n")
	self.file:flush()
end

function CSVLogger:log_value(field, value)
	assert(self.indices[field] ~= nil)

	local index = self.indices[field]
	self.cur_row[index] = tostring(value)
end

function CSVLogger:log_array(field, values)
	assert(self.indices[field] ~= nil)

	local index = self.indices[field]
	self.cur_row[index] = "\"" .. table.concat(values, ", ") .. "\""
end

function CSVLogger:flush()
	self.file:write(table.concat(self.cur_row, ", "), "\n")
	self.file:flush()

	for k = 1, #self.cur_row do
		self.cur_row[k] = "\"\""
	end
end

function CSVLogger:close()
	self.file:close()
end
