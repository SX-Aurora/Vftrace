# The user might have set some VFTR_ environment variables.
# After the test, we reset them to their original value.
for v in $(env | grep VFTR_)
do
  unset `echo $v | cut -f1 -d "="`
done

