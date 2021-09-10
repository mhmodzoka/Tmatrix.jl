# electromagnetics utility functions

Mu_vacuum = 1.25663706212e-6  # vacuum magnetic permeability
Eps_vacuum = 8.8541878128e-12  # vacuum electric permittivity

"""
    Convert between wavelength and frequency
    It should be similar to Matlab function "z_convert_wavelength_freq"
"""
function convert_wavelength_freq(
    wl_or_freq_input::R;
    input_unit = "",
    output_unit = "",
    decimals_to_round = nothing,
) where {R <: Real}
    c = 299792458
    # speed of light
    h_eV = 4.135667662e-15
    # [eV.s]

    input_unit = lowercase(input_unit)
    output_unit = lowercase(output_unit)

    # convert input to Hz
    if input_unit == "hz"
        freq_Hz = wl_or_freq_input

    elseif input_unit == "rad/s"
        freq_Hz = wl_or_freq_input / (2 * pi)

    elseif input_unit == "scufffrequnit"
        freq_radpers = wl_or_freq_input * 3e14
        freq_Hz = freq_radpers / (2 * pi)

    elseif input_unit == "nm"
        wl_m = wl_or_freq_input * 1e-9
        freq_Hz = c / wl_m

    elseif input_unit == "um"
        wl_m = wl_or_freq_input * 1e-6
        freq_Hz = c / wl_m

    elseif input_unit == "mm"
        wl_m = wl_or_freq_input * 1e-3
        freq_Hz = c / wl_m

    elseif input_unit == "cm"
        wl_m = wl_or_freq_input * 1e-2
        freq_Hz = c / wl_m

    elseif input_unit == "m"
        wl_m = wl_or_freq_input
        freq_Hz = c / wl_m

    elseif input_unit == "ev"
        wl_m = c / wl_or_freq_input * h_eV
        freq_Hz = c / wl_m

    elseif input_unit in ["cm^-1", "cm-1"]
        wl_m = 1 / wl_or_freq_input * 1e-2
        freq_Hz = c / wl_m

    else
        throw(
            DomainError("I don't know how to calculate for this `$input_unit` input unit"),
        )
    end

    # convert Hz into the desired unit
    if output_unit == "hz"
        wl_or_freq_output = freq_Hz

    elseif output_unit == "rad/s"
        wl_or_freq_output = freq_Hz * (2 * pi)

    elseif output_unit == "scufffrequnit"
        wl_or_freq_output = freq_Hz * (2 * pi) / 3e14

    elseif output_unit == "nm"
        wl_m = c / freq_Hz
        wl_or_freq_output = wl_m * 1e9

    elseif output_unit == "um"
        wl_m = c / freq_Hz
        wl_or_freq_output = wl_m * 1e6

    elseif output_unit == "mm"
        wl_m = c / freq_Hz
        wl_or_freq_output = wl_m * 1e3

    elseif output_unit == "cm"
        wl_m = c / freq_Hz
        wl_or_freq_output = wl_m * 1e2

    elseif output_unit == "m"
        wl_m = c / freq_Hz
        wl_or_freq_output = wl_m

    elseif output_unit == "ev"
        wl_m = c / freq_Hz
        wl_or_freq_output = c / wl_m * h_eV

    elseif output_unit in ["cm^-1", "cm-1"]
        wl_m = c / freq_Hz
        wl_or_freq_output = 1 / (wl_m * 1e2)

    else
        throw(
            DomainError(
                "I don't know how to calculate for this `$output_unit` output unit",
            ),
        )
    end

    if decimals_to_round !== nothing
        wl_or_freq_output = round(wl_or_freq_output, digits = Int(decimals_to_round))
    end

    return wl_or_freq_output
end

"""
    It needs one definition for permittivity (either Eps or Eps_r), and one definition for permeability (either Mu or Mu_r)
"""
function get_SpeedOfLight(; Eps = nothing, Mu = nothing, Eps_r = nothing, Mu_r = nothing)
    if (Mu == nothing) && (Mu_r !== nothing)
        Mu = Mu_r * Mu_vacuum
    else
        throw(DomainError("You must input either `Mu` or `Mu_r` keyword arguments."))
    end

    if (Eps == nothing) && (Eps_r !== nothing)
        Eps = Eps_r * Eps_vacuum
    else
        throw(DomainError("You must input either `Eps` or `Eps_r` keyword arguments."))
    end

    return 1 / sqrt(Mu * Eps)
end

"""
    get Refractive Index from electric Permittivity
"""
function get_RefractiveIndex_from_Permittivity(Eps_r)
    return sqrt(Eps_r)
end

"""
It needs one definition for permittivity (either Eps or Eps_r), and one definition for permeability (either Mu or Mu_r)
"""
function get_WaveVector(
    wl_or_freq_input;
    input_unit = "",
    Eps = nothing,
    Mu = nothing,
    Eps_r = nothing,
    Mu_r = nothing,
)
    Omega = convert_wavelength_freq(
        wl_or_freq_input;
        input_unit = input_unit,
        output_unit = "rad/s",
    )
    c = get_SpeedOfLight(; Eps = Eps, Mu = Mu, Eps_r = Eps_r, Mu_r = Mu_r)
    k = Omega / c
    return k
end
