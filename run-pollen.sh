# run script with nohup ./run-pollen.sh > pollen_out.txt &
# pollen_pg_stlm_script.jl
echo "Running Julia script pollen_pg_stlm_script.jl"
nohup /home/jrtipton/julia-1.7.1/bin/julia -t 32 pollen_pg_stlm_script.jl 
retval=$?

if [ $retval -ne 0 ]; then
  echo "Julia script pollen_pg_stlm_script.jl failed with $retval" 
  exit $retval
fi

# pollen_pg_stlm_overdispersed.jl
echo "Running Julia script pollen_pg_stlm_overdispersed_script.jl"
nohup /home/jrtipton/julia-1.7.1/bin/julia -t 32 pollen_pg_stlm_overdispersed_script.jl 
retval=$?

if [ $retval -ne 0 ]; then
  echo "Julia script pollen_pg_stlm_overdispersed_script.jl failed with $retval" 
  exit $retval
fi


# pollen_pg_stlm_latent.jl
echo "Running Julia script pollen_pg_stlm_latent_script.jl"
nohup /home/jrtipton/julia-1.7.1/bin/julia -t 32 pollen_pg_stlm_latent_script.jl 
retval=$?

if [ $retval -ne 0 ]; then
  echo "Julia script pollen_pg_stlm_latent_script.jl failed with $retval" 
  exit $retval
fi
