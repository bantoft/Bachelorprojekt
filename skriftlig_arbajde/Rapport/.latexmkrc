add_cus_dep('nlo', 'nls', 0, 'makenomenclature');

sub makenomenclature {
    my ($base_name) = @_;
    system "makeindex $base_name.nlo -s nomencl.ist -o $base_name.nls";
}

push @generated_exts, 'nlo', 'nls', 'ilg';
