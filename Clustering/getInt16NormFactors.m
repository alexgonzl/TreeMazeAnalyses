function INT_16Factors=getInt16NormFactors(fileName)

if verLessThan('matlab','9')
    addpath('../Lib/JSON/')
    HeaderInfo = loadjson(fileName);
else
    HeaderInfo = jsondecode(fileread(fileName));
end
fields = fieldnames(HeaderInfo);
x=strfind(fields,'Int_16');
int16field = fields(~cellfun('isempty',x));
INT_16Factors = HeaderInfo.(int16field{1});