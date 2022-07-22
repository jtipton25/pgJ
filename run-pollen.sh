# run script with ./run-pollen.sh > pollen_out.txt &
# pollen_pg_stlm_script.jl
echo "Fitting Julia script pollen_pg_stlm_script.jl"
# /home/jrtipton/julia-1.7.1/bin/julia -t 32 pollen_pg_stlm_script.jl > pollen_matern_out.txt
/home/jrtipton/julia-1.7.1/bin/julia -t 32 pollen_pg_stlm_script.jl 
retval=$?

if [ $retval -ne 0 ]; then
  # echo "Julia script pollen_pg_stlm_script.jl failed with $retval" > pollen_matern_err.txt
  echo "Julia script pollen_pg_stlm_script.jl failed with $retval" 
  exit $retval
fi

# pollen_pg_stlm_overdispersed.jl
echo "Fitting Julia script pollen_pg_stlm_overdispersed_script.jl"
# /home/jrtipton/julia-1.7.1/bin/julia -t 32 pollen_pg_stlm_overdispersed_script.jl > pollen_overdispersed_out.txt
/home/jrtipton/julia-1.7.1/bin/julia -t 32 pollen_pg_stlm_overdispersed_script.jl 
retval=$?

if [ $retval -ne 0 ]; then
  # echo "Julia script pollen_pg_stlm_overdispersed_script.jl failed with $retval" > pollen_overdispersed_err.txt
  echo "Julia script pollen_pg_stlm_overdispersed_script.jl failed with $retval" 
  exit $retval
fi


# pollen_pg_stlm_latent.jl
echo "Fitting Julia script pollen_pg_stlm_latent.jl"
# /home/jrtipton/julia-1.7.1/bin/julia -t 32 pollen_pg_stlm_latent_script.jl > pollen_latent_out.txt
/home/jrtipton/julia-1.7.1/bin/julia -t 32 pollen_pg_stlm_latent_script.jl 
retval=$?

if [ $retval -ne 0 ]; then
  # echo "Julia script pollen_pg_stlm_latent_script.jl failed with $retval" > latent_err.txt
  echo "Julia script pollen_pg_stlm_latent_script.jl failed with $retval" 
  exit $retval
fi
